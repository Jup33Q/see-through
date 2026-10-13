// SDXL-level UNet implemented with mlx-swift (random weights; perf/feasibility spike).
// Layout: NHWC throughout (MLX native). Structure mirrors diffusers UNet2DConditionModel
// as used by this project (see common/modules/layerdiffuse/layerdiff3d.py).

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - helpers

/// diffusers-style sinusoidal timestep embedding (flip_sin_to_cos = true, freq_shift = 0).
func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000) -> MLXArray {
  let half = dim / 2
  let idx = MLXArray((0..<half).map { Float($0) })
  let freqs = (idx * (-log(maxPeriod) / Float(half))).exp()
  let args = t.asType(.float32).reshaped(t.shape[0], 1) * freqs.reshaped(1, half)
  return MLX.concatenated([args.cos(), args.sin()], axis: -1)
}

func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
  let b = x.shape[0], h = x.shape[1], w = x.shape[2], c = x.shape[3]
  let x5 = x.reshaped(b, h, 1, w, 1, c)
  return MLX.broadcast(x5, to: [b, h, 2, w, 2, c]).reshaped(b, h * 2, w * 2, c)
}

func siluX(_ x: MLXArray) -> MLXArray { x * MLX.sigmoid(x) }

// MARK: - blocks

final class ResnetBlock: Module {
  @ModuleInfo var norm1: GroupNorm
  @ModuleInfo var conv1: Conv2d
  @ModuleInfo var timeProj: Linear
  @ModuleInfo var norm2: GroupNorm
  @ModuleInfo var conv2: Conv2d
  @ModuleInfo var convShortcut: Conv2d?

  init(inCh: Int, outCh: Int, tembCh: Int, groups: Int = 32) {
    _norm1 = .init(wrappedValue: GroupNorm(groupCount: groups, dimensions: inCh, eps: 1e-6, affine: true, pytorchCompatible: true))
    _conv1 = .init(wrappedValue: Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, padding: 1))
    _timeProj = .init(wrappedValue: Linear(tembCh, outCh))
    _norm2 = .init(wrappedValue: GroupNorm(groupCount: groups, dimensions: outCh, eps: 1e-6, affine: true, pytorchCompatible: true))
    _conv2 = .init(wrappedValue: Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 3, padding: 1))
    _convShortcut = .init(
      wrappedValue: inCh != outCh
        ? Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 1) : nil)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
    var h = x
    h = conv1(siluX(norm1(h)))
    let t = timeProj(siluX(temb)).reshaped(temb.shape[0], 1, 1, -1)
    h = h + t
    h = conv2(siluX(norm2(h)))
    if let sc = convShortcut { return sc(x) + h }
    return x + h
  }
}

/// Multi-head attention; self-attn when context == nil, cross-attn otherwise.
final class CrossAttention: Module {
  let heads: Int
  let headDim: Int
  @ModuleInfo var toQ: Linear
  @ModuleInfo var toK: Linear
  @ModuleInfo var toV: Linear
  @ModuleInfo var toOut: Linear

  init(dim: Int, heads: Int, headDim: Int, contextDim: Int? = nil) {
    let inner = heads * headDim
    self.heads = heads
    self.headDim = headDim
    _toQ = .init(wrappedValue: Linear(dim, inner))
    _toK = .init(wrappedValue: Linear(contextDim ?? dim, inner))
    _toV = .init(wrappedValue: Linear(contextDim ?? dim, inner))
    _toOut = .init(wrappedValue: Linear(inner, dim))
    super.init()
  }

  func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
    let ctx = context ?? x
    let b = x.shape[0], lq = x.shape[1], lk = ctx.shape[1]
    let q = toQ(x).reshaped(b, lq, heads, headDim).transposed(0, 2, 1, 3)
    let k = toK(ctx).reshaped(b, lk, heads, headDim).transposed(0, 2, 1, 3)
    let v = toV(ctx).reshaped(b, lk, heads, headDim).transposed(0, 2, 1, 3)
    let o = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: nil)
    return toOut(o.transposed(0, 2, 1, 3).reshaped(b, lq, heads * headDim))
  }
}

/// diffusers BasicTransformerBlock (norm_type=layer_norm): self-attn + cross-attn + GEGLU FF.
final class BasicTransformerBlock: Module {
  @ModuleInfo var norm1: LayerNorm
  @ModuleInfo var attn1: CrossAttention
  @ModuleInfo var norm2: LayerNorm
  @ModuleInfo var attn2: CrossAttention
  @ModuleInfo var norm3: LayerNorm
  @ModuleInfo var ffProj: Linear  // dim -> 2*inner
  @ModuleInfo var ffOut: Linear  // inner -> dim

  init(dim: Int, heads: Int, headDim: Int, contextDim: Int) {
    let inner = dim * 4
    _norm1 = .init(wrappedValue: LayerNorm(dimensions: dim, affine: true))
    _attn1 = .init(wrappedValue: CrossAttention(dim: dim, heads: heads, headDim: headDim))
    _norm2 = .init(wrappedValue: LayerNorm(dimensions: dim, affine: true))
    _attn2 = .init(wrappedValue: CrossAttention(dim: dim, heads: heads, headDim: headDim, contextDim: contextDim))
    _norm3 = .init(wrappedValue: LayerNorm(dimensions: dim, affine: true))
    _ffProj = .init(wrappedValue: Linear(dim, inner * 2))
    _ffOut = .init(wrappedValue: Linear(inner, dim))
    super.init()
  }

  func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
    var h = x + attn1(norm1(x))
    h = h + attn2(norm2(h), context: context)
    let proj = ffProj(norm3(h))
    let inner = proj.shape[2] / 2
    let xa = proj[0..., 0..., 0..<inner]
    let gate = proj[0..., 0..., inner..<(inner * 2)]
    return h + ffOut(xa * gelu(gate))
  }
}

/// diffusers Transformer2DModel wrapper: GroupNorm -> 1x1 conv in -> transformer blocks -> 1x1 conv out + residual.
final class SpatialTransformer: Module {
  @ModuleInfo var norm: GroupNorm
  @ModuleInfo var projIn: Conv2d
  @ModuleInfo var blocks: [BasicTransformerBlock]
  @ModuleInfo var projOut: Conv2d

  init(ch: Int, heads: Int, headDim: Int, depth: Int, contextDim: Int) {
    _norm = .init(wrappedValue: GroupNorm(groupCount: 32, dimensions: ch, eps: 1e-6, affine: true, pytorchCompatible: true))
    _projIn = .init(wrappedValue: Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1))
    _blocks = .init(
      wrappedValue: (0..<depth).map { _ in
        BasicTransformerBlock(dim: ch, heads: heads, headDim: headDim, contextDim: contextDim)
      })
    _projOut = .init(wrappedValue: Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1))
    super.init()
  }

  func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
    let b = x.shape[0], h = x.shape[1], w = x.shape[2], c = x.shape[3]
    var hs = projIn(norm(x)).reshaped(b, h * w, c)
    for blk in blocks { hs = blk(hs, context: context) }
    hs = hs.reshaped(b, h, w, c)
    return x + projOut(hs)
  }
}

// MARK: - down / up blocks

final class DownBlock: Module {
  @ModuleInfo var resnets: [ResnetBlock]
  @ModuleInfo var attentions: [SpatialTransformer]
  @ModuleInfo var downsample: Conv2d?

  init(
    inCh: Int, outCh: Int, tembCh: Int, numLayers: Int, hasAttn: Bool, heads: Int,
    headDim: Int, tfLayers: Int, contextDim: Int, addDownsample: Bool
  ) {
    var res: [ResnetBlock] = []
    var att: [SpatialTransformer] = []
    var ch = inCh
    for _ in 0..<numLayers {
      res.append(ResnetBlock(inCh: ch, outCh: outCh, tembCh: tembCh))
      if hasAttn {
        att.append(
          SpatialTransformer(
            ch: outCh, heads: heads, headDim: headDim, depth: tfLayers, contextDim: contextDim))
      }
      ch = outCh
    }
    _resnets = .init(wrappedValue: res)
    _attentions = .init(wrappedValue: att)
    _downsample = .init(
      wrappedValue: addDownsample
        ? Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 3, stride: 2, padding: 1)
        : nil)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, temb: MLXArray, context: MLXArray) -> (
    MLXArray, [MLXArray]
  ) {
    var h = x
    var skips: [MLXArray] = []
    for (i, res) in resnets.enumerated() {
      h = res(h, temb: temb)
      if !attentions.isEmpty { h = attentions[i](h, context: context) }
      skips.append(h)
    }
    if let d = downsample {
      h = d(h)
      skips.append(h)
    }
    return (h, skips)
  }
}

final class UpBlock: Module {
  @ModuleInfo var resnets: [ResnetBlock]
  @ModuleInfo var attentions: [SpatialTransformer]
  @ModuleInfo var upsample: Conv2d?

  init(
    inChannels: [Int], outCh: Int, tembCh: Int, hasAttn: Bool, heads: Int, headDim: Int,
    tfLayers: Int, contextDim: Int, addUpsample: Bool
  ) {
    _resnets = .init(wrappedValue: inChannels.map { ResnetBlock(inCh: $0, outCh: outCh, tembCh: tembCh) })
    _attentions = .init(
      wrappedValue: hasAttn
        ? inChannels.map { _ in
          SpatialTransformer(ch: outCh, heads: heads, headDim: headDim, depth: tfLayers, contextDim: contextDim)
        } : [])
    _upsample = .init(
      wrappedValue: addUpsample
        ? Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 3, padding: 1) : nil)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, skips: [MLXArray], temb: MLXArray, context: MLXArray) -> MLXArray
  {
    var h = x
    for (i, res) in resnets.enumerated() {
      h = MLX.concatenated([h, skips[i]], axis: -1)
      h = res(h, temb: temb)
      if !attentions.isEmpty { h = attentions[i](h, context: context) }
    }
    if let u = upsample {
      h = u(upsampleNearest2x(h))
    }
    return h
  }
}

// MARK: - SDXL UNet

/// SDXL base UNet topology at 1280px: latent 4x160x160, blocks 320/640/1280,
/// heads 10/20 (dh=64), transformer depths down=[1,2], mid=10, up=[10,2] (SDXL 1.0 config).
final class SDXLUNet: Module {
  let timeEmbedDim = 1280
  let contextDim = 2048

  @ModuleInfo var convIn: Conv2d
  @ModuleInfo var timeProj1: Linear
  @ModuleInfo var timeProj2: Linear
  @ModuleInfo var addProj1: Linear  // 1280 (text_embeds) + 6*256 (time_ids) = 2816 -> 1280
  @ModuleInfo var addProj2: Linear
  @ModuleInfo var downBlocks: [DownBlock]
  @ModuleInfo var midRes1: ResnetBlock
  @ModuleInfo var midAttn: SpatialTransformer
  @ModuleInfo var midRes2: ResnetBlock
  @ModuleInfo var upBlocks: [UpBlock]
  @ModuleInfo var convNormOut: GroupNorm
  @ModuleInfo var convOut: Conv2d

  override init() {
    let temb = 1280
    _convIn = .init(wrappedValue: Conv2d(inputChannels: 4, outputChannels: 320, kernelSize: 3, padding: 1))
    _timeProj1 = .init(wrappedValue: Linear(320, temb))
    _timeProj2 = .init(wrappedValue: Linear(temb, temb))
    _addProj1 = .init(wrappedValue: Linear(2816, temb))
    _addProj2 = .init(wrappedValue: Linear(temb, temb))

    // down: (320, no attn) -> (640, heads 10, depth 1) -> (1280, heads 20, depth 2); last block no downsample
    _downBlocks = .init(wrappedValue: [
      DownBlock(
        inCh: 320, outCh: 320, tembCh: temb, numLayers: 2, hasAttn: false, heads: 1, headDim: 1,
        tfLayers: 0, contextDim: 2048, addDownsample: true),
      DownBlock(
        inCh: 320, outCh: 640, tembCh: temb, numLayers: 2, hasAttn: true, heads: 10, headDim: 64,
        tfLayers: 1, contextDim: 2048, addDownsample: true),
      DownBlock(
        inCh: 640, outCh: 1280, tembCh: temb, numLayers: 2, hasAttn: true, heads: 20, headDim: 64,
        tfLayers: 2, contextDim: 2048, addDownsample: false),
    ])

    _midRes1 = .init(wrappedValue: ResnetBlock(inCh: 1280, outCh: 1280, tembCh: temb))
    _midAttn = .init(
      wrappedValue: SpatialTransformer(ch: 1280, heads: 20, headDim: 64, depth: 10, contextDim: 2048))
    _midRes2 = .init(wrappedValue: ResnetBlock(inCh: 1280, outCh: 1280, tembCh: temb))

    // up: SDXL skip-channel schedule
    _upBlocks = .init(wrappedValue: [
      UpBlock(
        inChannels: [1280 + 1280, 1280 + 1280, 1280 + 640], outCh: 1280, tembCh: temb,
        hasAttn: true, heads: 20, headDim: 64, tfLayers: 10, contextDim: 2048, addUpsample: true),
      UpBlock(
        inChannels: [1280 + 640, 640 + 640, 640 + 320], outCh: 640, tembCh: temb, hasAttn: true,
        heads: 10, headDim: 64, tfLayers: 2, contextDim: 2048, addUpsample: true),
      UpBlock(
        inChannels: [640 + 320, 320 + 320, 320 + 320], outCh: 320, tembCh: temb, hasAttn: false,
        heads: 1, headDim: 1, tfLayers: 0, contextDim: 2048, addUpsample: false),
    ])

    _convNormOut = .init(wrappedValue: GroupNorm(groupCount: 32, dimensions: 320, eps: 1e-5, affine: true, pytorchCompatible: true))
    _convOut = .init(wrappedValue: Conv2d(inputChannels: 320, outputChannels: 4, kernelSize: 3, padding: 1))
    super.init()
  }

  func callAsFunction(
    sample: MLXArray, timestep: MLXArray, encoderHiddenStates: MLXArray,
    textEmbeds: MLXArray, timeIds: MLXArray
  ) -> MLXArray {
    // time embedding: sinusoidal(320) -> MLP
    var temb = timestepEmbedding(timestep, dim: 320).asType(sample.dtype)
    temb = timeProj2(siluX(timeProj1(temb)))

    // added conditioning (SDXL text_time): text_embeds (1280) + Timesteps(time_ids) (6*256)
    var tids = timestepEmbedding(timeIds.reshaped(-1), dim: 256).reshaped(timestep.shape[0], -1)
    tids = tids.asType(sample.dtype)
    let addEmb = addProj2(siluX(addProj1(MLX.concatenated([textEmbeds, tids], axis: -1))))
    temb = temb + addEmb

    var h = convIn(sample)
    var skips: [MLXArray] = [h]
    for blk in downBlocks {
      let (out, sk) = blk(h, temb: temb, context: encoderHiddenStates)
      h = out
      skips.append(contentsOf: sk)
    }
    h = midRes1(h, temb: temb)
    h = midAttn(h, context: encoderHiddenStates)
    h = midRes2(h, temb: temb)
    for blk in upBlocks {
      let n = blk.resnets.count
      let sk = Array(skips.suffix(n))
      skips.removeLast(n)
      h = blk(h, skips: sk.reversed().map { $0 }, temb: temb, context: encoderHiddenStates)
    }
    h = convOut(siluX(convNormOut(h)))
    return h
  }
}
