# M3 — LayerDiff3D UNet CoreML（主体工程）报告

日期：2026-10-15 ｜ 机器：M5 Max / 128 GB / macOS 27 ｜ conda `see_through`（torch 2.14.1 MPS, coremltools 9.0, diffusers 0.37.0）｜ Swift 包：`swift/`（零外部依赖，`swift build -c release`）

## M3-1 合并单包 UNet 转换

产出 `out/models/layerdiff3d_unet_merged_fp16.mlpackage`（8.1 GB，不入库）——
M1 Spike B 的 body13/head11 两档包（各 7.6 GB、权重 99% 重复）合并为单包，
`EnumeratedShapes` 帧档 F∈{13,11}。消除 `group_index` Python 分支的做法
（roadmap 允许的薄 wrapper 的进一步简化）：

- 两处 group embedding（`layerdiff3d.py:1628` `group_embeds2` 作用于
  encoder_hidden_states、`:1689` `group_embeds` 作用于 text_embeds）的输入都是
  **静态量**（M2 embedding 包），因此 wrapper 直接置
  `unet.config.group_embedding=False`，改为传入预计算条件
  `ehs + group_embeds2[g](ehs)` / `te + group_embeds[g](te)`，图内零手术、
  数值与上游逐点等价（eager 对齐 spike-B 参考 PSNR 134 dB，两档同）。
- 预计算条件落盘 `out/statics/{body,head}/`（不入库），Swift 侧不需要
  group embedding 代码。

转换坑（均为 coremltools 9.0 在 EnumeratedShapes 符号形状下的限制）：

1. `layerdiff3d.py:1673` `timestep.repeat_interleave(num_frames)` 的 repeats
   符号化后 converter 的 `repeat_interleave` 要求常量（`Cannot add const [None]`）。
   trace 期 monkeypatch 不可靠（tracer 会把 repeats 包成 Tensor），最终方案：
   **torch.jit 图手术**——在 traced 子图上把该节点替换为
   `aten::expand(timestep, [Int(repeats)])`（`fix_repeat_interleave()`，
   最小复现验证 torch 侧手术前后逐 bit 一致、CoreML 两档形状均正确）。
2. `_group_norm_impl` 对符号 batch 构建 `[is0, 32, 10, 160, 160]` 混合
   python 列表触发 mb.reshape 拒绝。脚本内 monkeypatch 为 mb.concat 组装
   （`patch_coremltools_symbolic_group_norm()`，仅改 shape 组装，数值路径不变）。

校验（checkpoint 3，vs Spike B 冻结 fp32 参考）：

| 档 | PSNR | 首次前向 | 稳态前向（3 次） |
|---|---|---|---|
| body F=13 | 82.23 dB | 5.2 s | 5.4 s |
| head F=11 | 82.26 dB | 8.0 s | 4.9 s |

compute plan（ALL）：12057 ops = 6436 const + 5621 GPU + **0 ANE**（与 M1/M2
一致，ANE 优化留后续 spike）。转换耗时：trace 1453s + convert 91s。

## M3-2 golden 轨迹锚点

`golden_trajectory/{body,head}/`（1.2 GB，不入库）：30 步 ×
（unet_in/unet_out/latents_next）+ noise/latent_init/c_concat/vae_moments/
argb_in/page_alpha/prompt_embeds/decode_in + meta.json（timesteps、head 合成参数）。
**复现性验证：body 13 + head 11 共 24 层输出 PNG 与 M0 golden 逐 bit 一致**，
src_head.png 逐位一致。dump 用 M0 同路径（mps/bf16/seed=42）。

注意：锚点是 bf16 计算值（存 fp32）；`latent_dist.sample()` 的 VAE 采样走全局
RNG（seed_everything 覆盖），Swift 对齐模式用 `eps=(c_concat/0.13025-mean)/std`
反推（roundtrip max_abs 2.4e-7；该 checkpoint logvar 撞 -30 钳位，eps≈2400σ
属正常）。

## M3-3/4/5 Swift 调度器 + 胶水 + 逐阶段对齐

`swift/` 包（exe `ldrunner`）：NPY IO、DPMSolverMultistep
（sde-dpmsolver++/zero/leading/order2 精确移植）、CoreML 封装、PNG IO、
cv2 语义 resize、head 合成。调度器验证：timesteps 完全一致、**sigmas 与
diffusers 逐位一致（rel err 0）**（关键是逆向了 torch linspace f32 内核的
对称分段 FMA 与 cumprod 的 double 累乘）；30 步锚点 replay 反推噪声
σ∈[0.9989,1.0012]、零噪声闭合 2.4e-7（证明全部确定性/噪声系数正确）。
SDE 每步噪声 diffusers 用全局 RNG，Swift 对齐模式从锚点反推注入
（噪声与 model_output 无关，合法）。

对齐结果（merged 包，全部结果 JSON 在 `out/*.json`）：

| 阶段 | body | head |
|---|---|---|
| unet-step（单步 vs 锚点，step0/15） | 67.6 / 68.2 dB | 68.0 / 68.2 dB |
| denoise 30 步末步 latents | 56.8 dB | 53.4 dB |
| decode（锚点 decode_in → PNG vs golden） | 51.8–55.7 dB | 65.8–72.6 dB |
| e2e（src PNG → 编码 → 去噪 → 解码 → golden）min | **51.1 dB** | **65.1 dB** |

e2e 辅助指标：argb_in/page_alpha 逐位一致；c_concat 重建 151/154 dB；
VAE encoder moments vs 锚点 36.1/36.0 dB（CoreML fp16 vs torch bf16 的
真实差距，e2e 中被 eps 反推旁路不进最终误差）；误差主源为 UNet fp16 vs
torch bf16 的 30 步混沌放大（68 dB 单步 → ~55 dB 轨迹 → 像素域 51–65 dB），
全部远超「视觉一致」阈值（35 dB），目检与 golden 一致。

耗时（Swift 侧，含 MLMultiArray 封送）：UNet 前向 body avg 7.4s/step、
head avg 6.1s/step；30 步去噪 body 222.9s / head 183.9s（M0 MPS bf16 基线
两 pass 各 ~260s 含编解码；Python 侧独占 bench 5.4s/4.9s per step）。
进程峰值 RSS（转换期观测）：37 GB；推理期 ~16 GB。

## 复现

```
# conda env see_through；HF 权重需在缓存（HF_HUB_OFFLINE=1 防 hub 请求挂死）
ln -sf common/assets assets   # 若缺
python migration/m3/scripts/01_dump_trajectory.py        # 锚点（~8 min）
python migration/m3/scripts/02_convert_unet_merged.py    # 全量（eager+trace+convert+校验）
#   或 --from-traced 复用 out/traced_unet_merged.pt 跳过 trace
cd migration/m3/swift && swift build -c release
swift run -c release ldrunner sched-check-fixture
swift run -c release ldrunner sched-check --traj ../golden_trajectory/body
swift run -c release ldrunner unet-step --traj ../golden_trajectory/body --step 0
swift run -c release ldrunner denoise --traj ../golden_trajectory/body
swift run -c release ldrunner decode --traj ../golden_trajectory/body
swift run -c release ldrunner e2e --pass body   # head 同理
```

坑备忘：`MLModel(contentsOf:)` 在 macOS 27 需先 `MLModel.compileModel(at:)`；
HF hub 探测在无代理环境会挂死（用 HF_HUB_OFFLINE=1）；上游 PNG 落盘是截断
而非四舍五入；head/eyelash 一类小部件的 decode PSNR 对并发 GPU 负载敏感
（独占复测 58→73 dB）。
