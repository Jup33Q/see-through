// M1 Spike C: SDXL-level UNet forward on mlx-swift (Metal GPU). Random weights, fp16.
// Usage: unetbench [--iters N] [--batch B] [--res R]

import Foundation
import MLX
import MLXNN
import MLXRandom

func argValue(_ name: String, _ def: Int) -> Int {
  guard let i = CommandLine.arguments.firstIndex(of: name), i + 1 < CommandLine.arguments.count,
    let v = Int(CommandLine.arguments[i + 1])
  else { return def }
  return v
}

func mb(_ bytes: Int) -> String { String(format: "%.1f MB", Double(bytes) / 1e6) }

let iters = argValue("--iters", 3)
let batch = argValue("--batch", 1)
let res = argValue("--res", 160)

print("device: \(Device.defaultDevice()) (MLX default; gpu expected)")

// --- custom block checks (Transformer3D / GroupEmbedding feasibility) ---
runCustomBlockChecks()

// --- SDXL UNet ---
print("=== SDXL UNet forward (random weights, fp16) ===")
let t0 = CFAbsoluteTimeGetCurrent()
let model = SDXLUNet()

// cast all parameters to fp16 (deployment-realistic)
model.update(parameters: model.parameters().mapValues { _, v in v.asType(.float16) })
eval(model.parameters().flattenedValues())
let paramCount = model.parameters().flattenedValues().reduce(0) { $0 + $1.size }
let initDt = CFAbsoluteTimeGetCurrent() - t0
print(
  String(
    format: "params: %.2f B  init+materialize: %.2fs  weights(active): %@",
    Double(paramCount) / 1e9, initDt, mb(Memory.activeMemory)))

let latent = MLXRandom.normal([batch, res, res, 4]).asType(.float16)
let timestep = MLXArray([Float](repeating: 500, count: batch))
let encoder = MLXRandom.normal([batch, 77, 2048]).asType(.float16)
let textEmbeds = MLXRandom.normal([batch, 1280]).asType(.float16)
let timeIds = MLXRandom.normal([batch, 6])

@MainActor
func forward() -> MLXArray {
  model(
    sample: latent, timestep: timestep, encoderHiddenStates: encoder,
    textEmbeds: textEmbeds, timeIds: timeIds)
}

// warmup (includes Metal kernel compilation)
let tw = CFAbsoluteTimeGetCurrent()
let out0 = forward()
eval(out0)
print(
  String(
    format: "warmup forward (incl. compile): %.3fs  out shape: %@  dtype: %@",
    CFAbsoluteTimeGetCurrent() - tw, "\(out0.shape)", "\(out0.dtype)"))

var times: [Double] = []
for _ in 0..<iters {
  let t1 = CFAbsoluteTimeGetCurrent()
  let out = forward()
  eval(out)
  times.append(CFAbsoluteTimeGetCurrent() - t1)
}
let mean = times.reduce(0, +) / Double(times.count)
let best = times.min() ?? 0
print("per-forward times: \(times.map { String(format: "%.3f", $0) }) s")
print(String(format: "mean: %.3fs  best: %.3fs  batch=%d res=%dx%d", mean, best, batch, res, res))

let snap = Memory.snapshot()
print(
  "GPU memory: active=\(mb(snap.activeMemory)) peak=\(mb(snap.peakMemory)) cache=\(mb(snap.cacheMemory))"
)

// sanity: output is finite
let finite = (out0.asType(.float32).abs() .< Float.infinity).all().item(Bool.self)
print("output finite: \(finite)")
