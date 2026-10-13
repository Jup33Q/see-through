// Feasibility checks for the project's custom blocks in MLX Swift:
//  - CrossFrameAttn* / Transformer3DModel: batch-as-frame reshape + attention
//    (see common/modules/layerdiffuse/transformer3d.py CrossFrameTransformerBlock)
//  - GroupEmbedding: learnable [n_cls, dim] embedding broadcast-add + linear
//    (see common/modules/layerdiffuse/layerdiff3d.py)

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Port of diffusers GroupEmbedding: params [n_cls, dim]; x [n, s, d] -> x + params[:, None]; -> linear.
final class GroupEmbedding: Module {
  @ParameterInfo var params: MLXArray
  @ModuleInfo var linear: Linear

  init(nCls: Int, dim: Int) {
    _params = .init(wrappedValue: MLXRandom.normal([nCls, dim]) * 0.02)
    _linear = .init(wrappedValue: Linear(dim, dim))
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let p = x.ndim == 3 ? params.reshaped(params.shape[0], 1, params.shape[1]) : params
    return linear(x + p)
  }
}

/// Port of CrossFrameTransformerBlock's frame-mixing core: (B*F, S, C) <-> (B*S, F, C),
/// run attention over the F axis, then map back. Full block = this + FF_in/FF_out, all plain
/// Linear/LayerNorm — shown here with a single attention to prove the reshape+attn pattern.
final class CrossFrameAttention: Module {
  @ModuleInfo var normIn: LayerNorm
  @ModuleInfo var ffIn: Linear
  @ModuleInfo var norm1: LayerNorm
  @ModuleInfo var attn: CrossAttention

  init(dim: Int, heads: Int, headDim: Int) {
    _normIn = .init(wrappedValue: LayerNorm(dimensions: dim))
    _ffIn = .init(wrappedValue: Linear(dim, dim))
    _norm1 = .init(wrappedValue: LayerNorm(dimensions: dim))
    _attn = .init(wrappedValue: CrossAttention(dim: dim, heads: heads, headDim: headDim))
    super.init()
  }

  /// x: [B*F, S, C], attention across frames at each spatial token.
  func callAsFunction(_ x: MLXArray, batch: Int, frames: Int) -> MLXArray {
    let s = x.shape[1], c = x.shape[2]
    // (B*F, S, C) -> (B, F, S, C) -> (B, S, F, C) -> (B*S, F, C)
    var h = x.reshaped(batch, frames, s, c).transposed(0, 2, 1, 3).reshaped(batch * s, frames, c)
    let residual = h
    h = ffIn(normIn(h))
    h = h + residual
    h = attn(norm1(h)) + h
    // back: (B*S, F, C) -> (B, S, F, C) -> (B, F, S, C) -> (B*F, S, C)
    h = h.reshaped(batch, s, frames, c).transposed(0, 2, 1, 3).reshaped(batch * frames, s, c)
    return h
  }
}

func runCustomBlockChecks() {
  print("=== custom block feasibility checks ===")

  // 1) Cross-frame reshape semantics: must be exactly the PyTorch permute bijection.
  //    (B*F, S, C) -> (B*S, F, C) maps element (b*F+f, s, c) to (b*S+s, f, c).
  let (B, F, S, C) = (2, 3, 5, 4)
  let x = MLXArray((0..<(B * F * S * C)).map { Float($0) }).reshaped(B * F, S, C)
  let y = x.reshaped(B, F, S, C).transposed(0, 2, 1, 3).reshaped(B * S, F, C)
  let back = y.reshaped(B, S, F, C).transposed(0, 2, 1, 3).reshaped(B * F, S, C)
  eval(x, y, back)
  let xf = x.asArray(Float.self)
  let yf = y.asArray(Float.self)
  var maxErr: Float = 0
  for b in 0..<B {
    for f in 0..<F {
      for s in 0..<S {
        for c in 0..<C {
          let xi = ((b * F + f) * S + s) * C + c
          let yi = ((b * S + s) * F + f) * C + c
          maxErr = max(maxErr, abs(xf[xi] - yf[yi]))
        }
      }
    }
  }
  let backMatches = (back .== x).all().item(Bool.self)
  print(
    "cross-frame reshape: permute bijection maxErr=\(maxErr) roundtrip==identity: \(backMatches)")

  // 2) Cross-frame attention block end-to-end on realistic-ish dims.
  let cf = CrossFrameAttention(dim: 1280, heads: 20, headDim: 64)
  let hIn = MLXRandom.normal([B * F, 256, 1280])  // (B*F, S, C), S=16x16 tokens
  let hOut = cf(hIn, batch: B, frames: F)
  eval(hOut)
  print(
    "CrossFrameAttention: in \(hIn.shape) -> out \(hOut.shape) (frames mixed over F=\(F) axis)")

  // 3) GroupEmbedding: learnable [n_cls, dim] broadcast-add + linear (incl. 3-D broadcast case).
  let ge = GroupEmbedding(nCls: 11, dim: 1280)
  let g3 = ge(MLXRandom.normal([11, 77, 1280]))  // encoder_hidden_states path
  let g2 = ge(MLXRandom.normal([11, 1280]))  // text_embeds path
  eval(g3, g2)
  print("GroupEmbedding: [11,77,1280] -> \(g3.shape), [11,1280] -> \(g2.shape)")
}
