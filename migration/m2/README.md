# M2 — 静态化与小件 CoreML 迁移报告

日期：2026-10-15 ｜ 机器：M5 Max / 128 GB / macOS 27 ｜ conda `see_through`（torch 2.14.1 MPS, coremltools 9.0, diffusers 0.37.0, skimage 0.25.2）

## M2-1 静态 embedding 包

产出 `out/embeddings/`（15 MB，已入 git）：

- **LayerDiff v3**：24 个 tag（body 13 + head 11，硬编码列表见
  `diffusers_kdiffusion_sdxl.py:242-273`）× {`prompt_embeds` [1,77,2048],
  `pooled_prompt_embeds` [1,1280]}，外加 negative prompt `''` 的一组。
- **Marigold**：空 prompt embedding `empty_text_embed` [1,2,1024]
  （用 `24yearsold/seethroughv0.0.1_marigold` snapshot 自带的 text_encoder/tokenizer，
  不走 `prs-eth/marigold-depth-v1-1` 在线回退——GitHub/HF 直连不可达环境可复现）。
- **派生量**（存于 manifest，不占张量文件）：`add_time_ids = [1280,1280,0,0,1280,1280]`
  （`= (h,w,0,0,h,w)`，由分辨率唯一确定）、`guidance_scale=1.0`、30 步、1280px；
  Marigold 768px / DDIM 4 步。

格式：`manifest.json`（每 tensor 记录 file/shape/dtype/md5）+ 每 tensor 一个 `.npy`
（npy 头自带 shape/dtype，Swift 侧只需解析 ASCII 头 + 读原始 fp32 LE 数据）。

**精度口径**：编码用与 M0 golden 相同的 dtype/device（bf16 @ MPS），且调用的是上游
真实方法（`KDiffusionStableDiffusionXLPipeline.encode_cropped_prompt_77tokens` /
`MarigoldDepthPipeline.encode_empty_text`，通过未初始化实例直连，不走 from_pretrained
加载 4B UNet），因此与 M0 推理时 `cache_tag_embeds()` 缓存的值**逐 bit 一致**（同为 bf16
计算结果）；落盘为 fp32 是 bf16 的精确上转型，app 加载时自行 cast 到 fp16/bf16。
注意 `guidance_scale=1.0` ⇒ M0 golden 运行中 CFG 实际关闭
（`do_classifier_free_guidance = scale > 1`），negative embedding 为 M3+ 灵活性预留。

**加载验证**（`03_load_check.py`，纯 numpy、无 torch，模拟 app 侧消费）：
51 个 tensor 全部通过 md5/shape/dtype/有限性校验，tag 覆盖（24 + negative + marigold）
完整，且抽查三个不同 tag 的 embedding 互不相同。

## M2-2/3 小件 CoreML 转换 与 M2-4 对齐

正式包在 `out/models/`（`.mlpackage` 不入库）。对齐基准为 M1 Spike A 冻结的
fp32 参考张量（`spikes/m1/spike_a_vae/data/*.npy`；其 encoder 输入取自 golden
`reconstruction.png`/`face.png`，decoder 输入为固定 seed=1234 latent，即
「以 M0 golden 为源的模块级基准」）。阈值（记录于 `results.json`）：
图像域输出 PSNR[0,1] ≥ 35 dB 且 SSIM ≥ 0.99；latent 域输出 PSNR_dynamic ≥ 40 dB。

| 模块 | 包体积 | PSNR[0,1] (dB) | PSNR_dyn (dB) | SSIM | compute plan（ALL）| ANE 路径可跑 | 判定 |
|---|---|---|---|---|---|---|---|
| sdxl_vae_decoder | 99.2 MB | 64.61 | 73.05 | 1.0000 | 995 ops：590 const + 405 GPU + **0 ANE** | ✅（35.95 dB / SSIM 0.9955）| ✅ |
| sdxl_vae_encoder | 68.5 MB | 44.28 | 56.25 | —（latent 域）| 762 ops：457 const + 305 GPU + **0 ANE** | ✅（dyn 41.85 dB）| ✅ |
| trans_vae_decoder | 208.9 MB | 78.05 | 79.33 | 1.0000 | 3706 ops：2203 const + 1503 GPU + **0 ANE** | ✅（47.92 dB / SSIM 0.9994）| ✅ |
| trans_vae_encoder | 2.4 MB | inf | 96.44 | —（latent 域）| 84 ops：65 const + 19 GPU + **0 ANE** | ✅（dyn 41.15 dB）| ✅ |

- CoreML fp16 单次前向（computeUnits=ALL，5 次中位数）：448.9 / 389.3 / 382.2 / 5.5 ms，
  与 M1 Spike A 同量级；trans_vae_decoder 本次对齐（78.05 dB）优于 Spike A 记录的
  60.8 dB（Spike A 测量时同机有并行任务，此处为独占机器重测）。
- **ANE 路径可跑性**：每个包另以 `computeUnits=CPU_AND_NE` 加载并完成预测，且该路径下
  的对齐指标仍全部过阈值（见 `results.json` 的 `align_cpu_and_neural_engine`）。
- **compute plan 设备归属（M1 遗留要求）**：经 Xcode `coremlcompiler` 编译后用
  `load_compute_plan_from_path_on_device` 逐算子统计——4 个模块在 `ALL` 下的计算算子
  **100% preferred=GPU，0 个 NeuralEngine**，与 M1 UNet 的发现一致（fp16 大 conv/SDPA
  未被编译器判给 ANE）。ANE 优化（拆 attention + 量化）仍为独立后续 spike，不阻塞主线。
- 坑：coremltools 9 的枚举名是 `ComputeUnit.CPU_AND_NE`（不是 `CPU_AND_NEURAL_ENGINE`）；
  numpy 2.x 的 `np.bool` 不可 JSON 序列化（阈值判定需 `bool()` 显式转换）；
  `compute_plan` 必须对编译产物 `.mlmodelc` 调用（mlpackage 直读为 None），且
  `xcrun` 指向 CommandLineTools 无 coremlcompiler，须用 Xcode 工具链绝对路径。

## 复现

```
cd migration/m2/scripts
python 01_export_embeddings.py   # M2-1，自带 round-trip 校验
python 02_convert_models.py      # M2-2/3/4：转换 + 对齐 + compute plan + ANE 路径
python 03_load_check.py          # embedding 包纯 numpy 加载验收
```

依赖：M1 Spike A 的 `data/*.npy` 参考张量（缺失则先跑
`spikes/m1/spike_a_vae/scripts/01_make_reference.py`）。
