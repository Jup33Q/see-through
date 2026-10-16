# M4 — Marigold UNet CoreML（伪深度估计）报告

日期：2026-10-16 ｜ 机器：M5 Max / 128 GB / macOS 27 ｜ conda `see_through`（torch 2.14.1 MPS, coremltools 9.0, diffusers 0.37.0）｜ Swift 侧并入 M3 包 `migration/m3/swift/`（ldrunner 新增 6 个 `mari-*` 子命令）

范围：apply_marigold 全链路（img_list 装配 → pad_rgb_torch → Marigold VAE enc ×21 →
UNetFrameConditionModel 4 步 DDIM（v-prediction）→ VAE dec ×20 → 深度后处理 →
23 张 `*_depth.png`）。帧数固定 F=20（19 个 VALID_BODY_PARTS_V2 槽位 + fullpage），
768px ⇒ latent 96²。与 M3 的差异：config 无 `group_embedding`（两处分支自然跳过）、
静态单档形状（无 EnumeratedShapes）、DDIM eta=0 全确定性（无 SDE 噪声注入/反推）。

## M4-1 golden 轨迹锚点

`golden_trajectory/`（616 MB，不入库）：直接 hook 并调用真实
`utils.inference_utils.apply_marigold`（mps/bf16/seed=42/resolution=768/steps=-1→4，
输入为 golden 中间 PNG 的 scratch 拷贝）：
argb_pre_pad(21,4,768,768)、vae_rgb_in(21,3,768,768)、vae_latents(21,4,96,96)、
cond_latent(20,8,96,96)、latent_init、4 步 × (unet_in/unet_out/latents_next)、
decode_in/decode_out、depth_pred_768、depth_pred_src(20,1280,1280)、
empty_text_embed、meta.json（timesteps [999,749,499,249]、scheduler config 全量）。
**复现性验证：23/23 张 `*_depth.png` 与 M0 golden 逐 bit 一致**（推理墙钟 12.3s）。

## M4-2 UNet 转换

`out/models/marigold_unet_fp16.mlpackage`（2.43 GB，1.21B 参数，不入库）。
inputs: sample(1,20,12,96,96) / timestep(1,) / encoder_hidden_states(20,2,1024)。
静态形状 ⇒ M3 的两个符号补丁（repeat_interleave 手术、_group_norm_impl）均不需要，
wrapper 零语义改动（无 added_cond_kwargs）。

校验（vs 锚点 bf16 参考）：eager fp32 step0/2 = 61.78/63.95 dB（fp32↔bf16 底噪），
CoreML fp16 step0/2 = **61.72/63.93 dB**（阈值 60，贴 eager 底噪 ⇒ fp16 几乎无损）。
稳态前向 1.29s（Python bench，首次 3.2s）。转换耗时：trace 6666s + convert 307s。
compute plan（ALL）：4821 ops = 2689 const + 2132 GPU + **0 ANE**（与 M1–M3 一致，
ANE 优化留独立 spike）。

## M4-3 Marigold VAE enc/dec 转换

SD2 系 AutoencoderKL（scaling 0.18215，与 M2 SDXL VAE 不同，必须单转）。
`marigold_vae_encoder_fp16.mlpackage`（68.5 MB）：rgb(1,3,768,768)→mean*0.18215；
`marigold_vae_decoder_fp16.mlpackage`（99.2 MB）：z/0.18215→decoder→mean(dim=1)。
对齐（锚点帧 0/19，dyn PSNR 阈值 40）：enc 65.98/60.81 dB、dec 55.04/57.16 dB；
ANE 路径（CPU_AND_NE）可跑且仍过阈值（enc 63.2 / dec 52.2 dB）。
前向 91.7ms / 126.6ms；compute plan 全 GPU、0 ANE（756/989 ops）。

## M4-4/5 Swift DDIM 调度器 + 胶水 + 逐阶段对齐

`DDIMScheduler.swift`（diffusers 0.37.0 scheduling_ddim.py 精确移植，config：
scaled_linear 0.00085–0.012、**rescale_betas_zero_snr**（euler 版算法 1，全 f32
逐步舍入）、trailing、steps_offset=1、set_alpha_to_one=false、v_prediction、
eta=0）：timesteps 与锚点完全一致，alphas_cumprod 与 diffusers max_rel_err
4.4e-16（903/1000 逐位），fixture 4 步 replay **逐位一致（max_abs=0）**。

胶水（`MarigoldCommands.swift`）：img_list 装配（alpha<15→0、eyes/hair 合成、
blended_alpha 截断）、pad_rgb_torch（dk=1.2 alpha 金字塔 + torch bilinear
align_corners=False 语义）、cv2 语义 float resize、深度后处理
（截断落盘 + compose tag 的 mask_invis/median 路径，实测 eyes/hair 真实触发）。

对齐结果（全部 JSON 在 `out/*.json`，汇总 `swift_alignment_summary.json`）：

| 阶段 | 结果 |
|---|---|
| sched-check-fixture | PASS（replay 逐位） |
| sched-check 锚点 4 步 | PASS（max_abs ≤0.031，bf16 量化量级，阈值 0.05） |
| unet-step 0/2 | **61.72 / 63.93 dB**（阈值 60） |
| denoise 4 步末步 vs decode_in | 54.8 dB |
| decode（锚点 decode_in → 23 PNG vs golden）min | **48.24 dB**（阈值 35） |
| e2e（golden PNG → 装配 → 编码 → 去噪 → 解码 → golden）min | **48.23 dB**，目检一致 |

e2e 辅助指标：argb_pre_pad 119.1 dB（2–3 像素 1/255 级，cv2 INTER_AREA 8U 定点
vs M3 浮点面积平均的舍入差）、vae_rgb_in 54.2 / vae_latents 57.8 /
cond_latent 54.1 dB（锚点 bf16 量化量级）。误差主源同 M3：fp16 CoreML vs bf16
锚点的逐步累积；48 dB 远超视觉一致阈值。

耗时（Swift 侧，含封送）：UNet avg 1.47s/step（4 步合计 5.9s），VAE dec ×20
合计 2.6s，**e2e 全程约 16s**（vs M0 MPS bf16 基线 marigold 阶段 35s，约 2.2×，
且 M0 的 35s 不含 layerdiff）。内存：转换期 RSS 峰值 19.4 GB；推理期 ~7.7 GB。

## 复现

```
# conda env see_through；HF_HUB_OFFLINE=1；ln -sf common/assets assets（若缺）
python migration/m4/scripts/01_dump_trajectory.py   # 锚点（~1 min，23/23 逐bit验证）
python migration/m4/scripts/02_convert_unet.py      # trace+convert+校验（数小时；--from-traced 可复用 traced_unet.pt）
python migration/m4/scripts/03_convert_vae.py       # VAE enc/dec（~10 min）
python migration/m3/swift/Fixtures/gen_ddim_fixtures.py  # DDIM fixtures（已入库）
cd migration/m3/swift && swift build -c release
swift run -c release ldrunner mari-sched-check-fixture
swift run -c release ldrunner mari-sched-check
swift run -c release ldrunner mari-unet-step --step 0   # 或 --step 2
swift run -c release ldrunner mari-denoise
swift run -c release ldrunner mari-decode
swift run -c release ldrunner mari-e2e
```

坑备忘：trace/convert 期间勿并行跑其他重负载（首次 trace 与 VAE 转换互抢 CPU
导致超时；checkpoint3 出现过一次 900s 的异常 predict，系夜间系统负载）；
M4 VAE decoder 输出是 fp16 MLMultiArray（CoreMLWrap.toFloatArray 已支持）；
JSONSerialization 遇 ±inf PSNR 抛 NSException（jsonWriteBase 已清洗）；
DDIM trailing 在 diffusers 0.37 是 `round(arange(1000,0,-250))-1`=[999,749,499,249]，
勿按旧版 leading+offset 推。
