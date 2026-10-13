# M1 Spike A：VAE 小件 CoreML 可行性报告

日期：2026-10-13 ｜ 机器：M5 Max / 128 GB / macOS 27 ｜ coremltools 9.0, torch 2.14.1 (MPS), diffusers 0.37.0

## 结论：**通过决策门（PSNR ≥ 35 dB），4/4 模块全部转换成功**

SDXL VAE decoder / encoder、TransparentVAE decoder（UNet1024）/ encoder（LatentTransparencyOffsetEncoder）
均经 `torch.jit.trace` → coremltools `mlprogram`（fp16, `computeUnits=ALL`）转换成功，与 PyTorch fp32
参考输出（MPS）对齐误差全部在 PyTorch fp16 自身底噪的量级上或附近。CoreML 单次前向均快于
PyTorch MPS fp32 参考。可以进入 M1 下一阶段（更大件 / 端到端集成）。

## 数据表

输入规格：latent 4×160×160（即 1280×1280 图像档），固定 seed=1234；encoder 输入取自 golden
`reconstruction.png` / `face.png`（缩放到 1280）。参考为 PyTorch fp32 on MPS。

| 模块 | 转换 | trace (s) | convert (s) | 包体积 | max-abs-diff | PSNR[0,1] (dB) | PSNR_dyn (dB) | fp16 底噪 PSNR[0,1] | CoreML 前向 | Torch fp32 MPS |
|---|---|---|---|---|---|---|---|---|---|---|
| sdxl_vae_decoder | ✅ | 97.0 | 2.0 | 99.2 MB | 0.0189 | **64.6** | 73.1 | 64.7 | 432 ms | 996 ms |
| sdxl_vae_encoder | ✅ | 52.8 | 1.7 | 68.5 MB | 9.56 | **44.3** | 56.2 | 44.5 | 378 ms | 573 ms |
| trans_vae_decoder | ✅ | 28.0 | 6.6 | 208.9 MB | 0.134 | **60.8** | 63.0 | 77.0 | 379 ms | 980 ms |
| trans_vae_encoder | ✅ | 2.3 | 0.3 | 2.4 MB | 12.75 | inf¹ | **96.4** | inf¹ | 5.1 ms | 13.9 ms |

- **PSNR[0,1]**：双方输出 clamp 到 [0,1] 图像域后计算（决策门指标，门限 35 dB）。
- **PSNR_dyn**：以 `max|ref|` 为峰值的 PSNR，用于 latent 域输出（moments / offset），这些输出
  数值范围远超 [0,1]（sdxl_enc absmax≈30.3，trans_enc offset×300 后 absmax≈17563），clamp 会
  把 PSNR[0,1] 压成无意义值¹（trans_enc 的输出 clamp 后两侧一致饱和 → inf）。
- **fp16 底噪**：同模块 PyTorch fp16 (MPS) vs fp32 (MPS)，用于界定「fp16 精度策略本身」的误差下限。
- 耗时为 5 次中位数（warmup 2）。**注意：同机有其他 spike 任务并行运行，绝对耗时仅供参考**；
  相对比较（CoreML vs Torch 同机同窗测量）仍有效。

## 逐模块判读

- **sdxl_vae_decoder**：CoreML fp16 与 PyTorch fp16 底噪几乎完全一致（max-abs 0.0189 vs 底噪
  0.0194；PSNR 64.6 vs 64.7）→ 误差 100% 来自 fp16 本身，转换无附加损失。输出为图像域，最具代表性。
- **sdxl_vae_encoder**：同样贴着 fp16 底噪（44.3 vs 44.5 dB）。moments 的 logvar 通道数值大，
  max-abs 9.56 对应 dynamic PSNR 56 dB，可接受。img2img 路径可用。
- **trans_vae_decoder**：PSNR 60.8 dB 远超门限，但误差比 PyTorch fp16 底噪高 ~20×（max-abs 0.134
  vs 0.0062）。UNet1024 的注意力块 + 长残差链在 CoreML fp16 中间张量cast 策略下损失更大。
  若后续端到端对齐需要更紧，可对该模块试 `compute_precision=FLOAT32` 或只对注意力子图保 fp32。
- **trans_vae_encoder**：纯 Conv+SiLU 小网，转换秒级完成，dynamic PSNR 96 dB，CoreML 5.1 ms。

## ANE 证据（证据级别：API + 工具链级，非逐 op 实测）

- 转换与加载均指定 `computeUnits=ALL`，模型格式为 `mlProgram`（ANE eligible op set）。
- Xcode 工具链 `coremlcompiler compile` 对 4 个 mlpackage 全部编译通过
  （路径：`/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/coremlcompiler`；
  注意 `xcrun` 指向 CommandLineTools，找不到 coremlcompiler，须用 Xcode 内绝对路径）。
- 性能旁证：trans_vae_encoder 5.1 ms、各模块均快于 MPS fp32，与加速器执行特征一致。
- 未做 Instruments/os_signpost 逐 op ANE 归属 profiling（需 GUI profiling 会话，超本 spike 时间盒）；
  如需硬性 ANE 占比证据，建议 M1 后续用 Instruments 的 Core ML template 补一次。

## 踩坑与对策

1. **diffusers 0.37 的 `Decoder.forward` 直接返回 Tensor**（不是带 `.sample` 的 dataclass），wrapper
   里做了 `hasattr(out, "sample")` 兼容。trace 前务必先单跑一次 wrapper 验证。
2. **trace 大头在 VAE decoder**（97 s，mid attention 在 160×160 latent 上序列长度 25600）。CPU fp32
   trace 内存峰值高但 128 GB 无压力。convert 本身都只要秒级。
3. `torch.jit.trace` 有 FutureWarning（官方推 `torch.export`），coremltools 9.0 对 trace 路线支持完好，
   本 spike 未踩 op 不支持问题（SDPA/GroupNorm/attention 均正常转换）。
4. latent 域输出的 PSNR 判读必须用 dynamic-peak 口径，clamp-[0,1] 口径对 encoder 输出无意义。
5. 并行 spike 会污染绝对耗时；如需精确 benchmark 需独占机器重测。

## 产物

- 参考张量：`data/*.npy`（输入 + fp32 参考输出）
- 模型：`models/{sdxl_vae_decoder,sdxl_vae_encoder,trans_vae_decoder,trans_vae_encoder}_fp16.mlpackage`（共 361 MB）
- 指标：`results.json` ｜ 日志：`logs/01_reference.log`, `logs/02_*.log`
- 复现：`cd scripts && python 01_make_reference.py && python 02_convert_eval.py`（需先激活 see_through env）
