# See-through → macOS 原生迁移 Roadmap（SwiftUI + CoreML ANE 优先）

> 本文件是固化的执行路线图。新会话通过 `docs/ACTIVATION_PROMPT.md` 激活，
> 按本文档的当前阶段执行，完成后更新文末状态表并提交。
>
> 目标：把 See-through（SIGGRAPH 2026，单图动漫角色 23 层分解 → PSD）从
> PyTorch/CUDA 研究代码迁移为 macOS 原生 SwiftUI 应用，推理后端 CoreML（ANE 优先），
> 受阻模块用 mlx-swift 兜底（仍为纯 Swift），Python 桥接仅为最后备选。

- 上游：`shitagaki-lab/see-through`；本 fork：`Jup33Q/see-through`
- 目标机：Apple Silicon（开发机 M5 Max / 128 GB / macOS 27）
- 创建：2026-10-13

---

## 0. 技术事实底账（探索结论，改动代码前先读这里）

主推理路径 `inference/scripts/inference_psd.py` 三段式：

| 阶段 | 实现 | 模型（HF repo） | 精度 |
|---|---|---|---|
| 1. 透明层生成 | `common/utils/inference_utils.py:29` `apply_layerdiff` | `layerdifforg/seethroughv0.0.2_layerdiff3d` | bf16 硬编码 |
| 2. 伪深度估计 | `inference_utils.py:189` `apply_marigold` | `24yearsold/seethroughv0.0.1_marigold` | bf16 硬编码 |
| 3. 后处理+PSD | `inference_utils.py:436` `further_extr` → `dump_parts_psd` | 无（纯 CPU，psd-tools） | — |

模型组件清单（原生迁移面，共 6 组网络 + 后处理）：

1. **LayerDiff3D UNet**（`common/modules/layerdiffuse/layerdiff3d.py:560` `UNetFrameConditionModel`）
   — SDXL 骨架（320/640/1280）+ `CrossFrameAttn*` + `Transformer3DModel` 帧块
   （`transformer3d.py`，tag 当 frame）+ `GroupEmbedding`（`layerdiff3d.py:533`）。
   两次调用：body pass 13 帧（group 0）、head pass 11 帧（group 1）。**核心迁移风险点。**
2. **SDXL VAE**（`AutoencoderKL`，img2img 需要 enc + dec）。
3. **TransparentVAE**（`common/modules/layerdiffuse/vae.py:325`）：
   `LatentTransparencyOffsetEncoder`（小 CNN）+ `UNet1024` 解码器，出 premultiplied RGBA。
4. **CLIP 双文本编码器**：推理只对固定 24 个 prompt 各算一次 embedding 后卸载
   （`diffusers_kdiffusion_sdxl.py:242-273`，v3：body 13 + head 11，tag 列表在该函数内硬编码）
   → **原生版不迁移 CLIP，改为随 app 分发预计算 embedding 静态文件**。
   Marigold 同理只有一个空 prompt embedding。
5. **Marigold UNet**：与 1 同一个 `UNetFrameConditionModel` 类，768px / DDIM 4 步。
6. **调度器**：LayerDiff 用 DPMSolverMultistep（`sde-dpmsolver++`, `final_sigmas_type=zero`），
   Marigold 用 DDIM——都是简单数学，Swift 重写。
7. **后处理**：连通域、左右拆分、深度中值排序、PSD 写出（`io_utils.py:498-519` psd-tools）。
   LaMa inpainting 仅在 `--tblr_split` 且存在 hair 层时触发，默认路径不涉及，**暂缓迁移**。

明确排除项：NF4 量化脚本（bitsandbytes，CUDA-only）、blockswap 脚本、SAM/SAM2/detectron2/mmdet
（仅数据管线与 demo，不在 `inference_psd.py` 路径）、`ui/`（PyQt6 标注工具，与推理无关）。

CUDA 硬编码点（MPS 移植面，约 10 行）：
`common/utils/inference_utils.py:65-71,195`（`.to('cuda')` + group_offload）；
无害但需 guard 的 `torch.cuda.empty_cache()`（`diffusers_kdiffusion_sdxl.py:273`、
`marigold_depth_pipeline.py:408,471`）、`torch.cuda.manual_seed_all`（`torch_utils.py:287`）。

---

## M0 — 基线部署（PyTorch MPS）+ golden 输出冻结

目的：跑通端到端，冻结逐层 golden 输出作为后续所有迁移阶段的数值对齐基准。

任务清单：
1. `conda create -n see_through python=3.12`，装 torch≥2.9（MPS 自带）；`requirements.txt`
   去掉 cu128 pin 后安装（`-e ./common` 必须，包名 `live2d-common`）。
2. 设备移植：加 `--device` 参数（默认 auto：cuda→mps→cpu），bf16 改按设备选 dtype
   （MPS 先试 bf16，GroupNorm/SiLU 有坑则回退 fp32 开关）；给 `torch.cuda.*` 调用加
   `torch.cuda.is_available()` guard。
3. 首次运行自动从 HF 下载权重（约 8–10 GB）。
4. 跑 `python inference/scripts/inference_psd.py --srcp assets/test_image.png --save_to_psd`，
   输出到 `workspace/layerdiff_output/`。
5. **冻结 golden**：把 `test_image` 的全部中间产物（每 tag RGBA PNG、`*_depth.png`、
   `reconstruction.png`、`info.json`、最终 PSD 及其 `.psd.json`）复制到
   `tests/golden/test_image/`（加入 git 或单独存放，在状态表记录位置）。
6. 记录基线：各阶段耗时、峰值内存（`/usr/bin/time -v` 或活动监视器）。

验收：PSD 含 23 个语义层、可在 Photoshop/psd-tools 正常打开；golden 输出归档完成；
基线性能数据写入状态表。

风险：transformers 5.0.0 / diffusers 0.37.0 与 torch 2.9 的组合问题；MPS bf16 数值异常
（回退 fp32 再诊断）；HF 下载慢（可配镜像）。

---

## M1 — 纯 Swift 可行性 spike（架构决策门）

目的：用证据决定推理后端路线，**在写 app 之前完成**。

- Spike A（小件）：coremltools 转换 SDXL VAE decoder + TransparentVAE enc/dec →
  `computeUnits=.all` 跑 ANE，与 golden 对齐（PSNR≥35dB 视为可继续，不足则查 fp16 精度策略）。
- Spike B（核心）：`UNetFrameConditionModel` CoreML 转换。固定 shape trace
  （1280 分辨率 latent 160²、body 13 帧 / head 11 帧两档），检查：
  frame 批次 reshape/Transformer3D/GroupEmbedding 的 trace 兼容性、
  SDPA → coremltools 映射、ANE 编译落盘率、单次前向耗时与内存。
  参考 apple/ml-stable-diffusion 的 SDXL UNet 切分/attention 处理手法。
- Spike C（兜底）：mlx-swift 跑通一个 SDXL 级 UNet 前向，确认自定义块可在 MLX 实现。

决策门：
- UNet 能转 CoreML → **路线 A：全 CoreML（ANE 优先）**。
- UNet 转不了但 mlx-swift 可行 → **路线 B：CoreML（VAE/小件）+ mlx-swift（UNet）**，仍纯 Swift。
- 都不行 → 路线 C：SwiftUI + 内嵌 Python 后端（Process 桥接），并在状态表记录阻塞原因。

验收：三个 spike 各有结论记录（转换脚本、日志、对齐数据），决策门选择写入状态表。

---

## M2 — 静态化与小件 CoreML 迁移

1. 从 PyTorch 侧导出静态 embedding 包：24 个 tag 的 prompt_embeds/pooled_prompt_embeds +
   Marigold 空 prompt embedding + `add_text_embeds`/`add_time_ids` 所需派生量，
   存为二进制（带 shape/dtype 头），随 app 分发。**运行时不再需要 CLIP。**
2. SDXL VAE enc/dec → CoreML `.mlpackage`。
3. TransparentVAE enc/dec → CoreML。
4. 每个模块独立对齐 golden（PSNR/SSIM 阈值记录在案）。

验收：小件全部 ANE 可跑且对齐达标；embedding 包加载验证。

---

## M3 — LayerDiff3D UNet CoreML（主体工程）

前置资产：M1 Spike B 已验证转换链路（`spikes/m1/spike_b_unet/convert_unet.py`，
body13/head11 两档 trace→convert→对齐 PSNR≈82 dB 全通，mlpackage 留本地）；
文本条件用 M2 embedding 包（`migration/m2/out/embeddings/`，运行时无 CLIP）；
HF 缓存 2026-10-15 被清理，开工前需重下权重（`HTTPS_PROXY=http://127.0.0.1:7890`
或 `HF_ENDPOINT=https://hf-mirror.com`，约 8–10 GB）。

1. UNet 正式转换：复用 Spike B 链路，固定 shape（latent 160²、body 13 / head 11 帧）
   → fp16。**合并两档为单包**：`group_index` 在 `layerdiff3d.py:1628,1689` 是 Python
   索引（trace 时常量化），致两档包 99% 权重重复（各 7.6 GB）——用薄 wrapper 把 group
   embedding 选择改为张量入参（不改上游语义），合并后单包约 7.6 GB；必要时 palettize。
2. 对齐锚点 dump：M0 golden 只冻了最终 PNG，需先在 PyTorch 侧 dump body/head 两 pass
   的 30 步去噪轨迹（timesteps/sigmas/每步 latents/单步 UNet 输入输出对）作为调度器
   与轨迹对齐基准，存 `migration/m3/golden_trajectory/`。
3. DPMSolverMultistep（sde-dpmsolver++，zero final sigmas）调度器 Swift 实现，
   按锚点 2 逐步对齐 sigma/latent。
4. 管线胶水 Swift 化：源图 latent 编码（复用 M2 CoreML 小件）、c_concat 拼装与帧扩展、
   固定 seed 初始噪声。**注意：golden 运行 cfg=1.0 ⇒ CFG 关闭**
   （`do_classifier_free_guidance = scale > 1`），默认路径无 CFG 双批/uncond 分支，
   不要实现多余路径（negative embedding 已导出备用）。
5. 逐阶段对齐：单步 UNet 输出 → 完整去噪轨迹 → 解码 PNG。

验收：body/head 两 pass 输出 PNG 与 golden 视觉一致、逐层 PSNR 达标；记录 CoreML
耗时/内存（已知 computeUnits=ALL 下计算算子 100% 落 GPU、ANE 落盘率 0%——
ANE 优化属独立后续 spike，不阻塞本阶段验收）。

## M4 — Marigold UNet CoreML

复用 M3 全部机制（同一 UNet 类），768px / DDIM 4 步；对齐 `*_depth.png` golden。

## M5 — 后处理与 PSD 导出 Swift 化

1. `further_extr` 移植：连通域、左右拆分、深度中值排序 → Swift（Accelerate/vImage）。
2. 原生 PSD writer：8-bit RGBA 分层 + RLE 压缩，层序按深度中值（对齐 `dump_parts_psd`）；
   用 psd-tools 读回 + Photoshop 打开双向验证。
3. （可选）`*_depth.psd` 与 `.psd.json` 旁车产物。

验收：Swift 端到端产出 PSD 与 M0 golden PSD 结构/层名/顺序一致。

## M6 — SwiftUI App 整合

1. macOS app：拖入图片 → 分段进度（layerdiff body/head、marigold、后处理）→
   分层预览（可开关各层）→ 导出 PSD。
2. 模型包管理：HF 权重下载 + CoreML 转换产物的本地缓存与校验。
3. Benchmark 页/日志：ANE 路径 vs M0 MPS 基线的端到端耗时与峰值内存。

验收：从拖图到导出 PSD 全流程在 app 内完成；性能数据写入状态表。

---

## 状态表（每次激活会话结束时更新并提交）

| 阶段 | 状态 | 日期 | 备注 |
|---|---|---|---|
| M0 基线部署+golden | ✅ | 2026-10-13 | MPS bf16 端到端跑通。基线(M5 Max):总耗时 580s(layerdiff body 4m16s / head 4m22s / marigold 35s / 后处理+PSD 5s),峰值 RSS 23.6 GiB,peak footprint 51 GB。PSD 17 层(test_image 实际非空部件数;23 为全 tag 上限,headwear/earwear/neckwear/tail/wings/objects 为空被丢弃,head 为中间产物),psd-tools 打开+composite 验证通过。golden 冻结于 `tests/golden/test_image/`(54 文件 20 MB,已入 git)。环境:conda `see_through` py3.12,torch 2.14.1;scipy 升 1.18.1(1.15.3 wheel 与 macOS 27 不兼容);requirements 中 timm/convnext_perceptual_loss 两个 git pin 与推理无关且 GitHub 直连不可达,跳过(未装)。改动:`--device/--dtype` 参数 + cuda guard,见 M0 任务 2 |
| M1 可行性 spike + 决策门 | ✅ | 2026-10-13 | **决策门：路线 A（全 CoreML）**。Spike A：SDXL VAE enc/dec + TransparentVAE enc/dec 4/4 转换成功，PSNR 44.3–64.6 dB（门限 35），全部贴 fp16 底噪。Spike B：UNetFrameConditionModel（4.07B）零修改 trace→convert 全链路通过，body13/head11 两档 @latent 160² 编译落盘（各 7.6 GB mlpackage），对齐 PSNR≈82 dB，稳态前向 5.1s/4.3s（vs MPS fp32 19.3s/16.2s，同机并行下约 3.8×）；SDPA 被 coremltools 9.0 原生映射，三个风险点（frame reshape/Transformer3D/GroupEmbedding）最小复现全过。遗留：ANE 落盘率 0%（4754 算子全落 GPU），上 ANE 需拆 attention+量化，留作后续 spike；group_index 为 Python 分支致两档包权重 99% 重复，M3 可改张量入参合并。Spike C（兜底验证）：mlx-swift 跑通 SDXL 级 UNet 前向（1.99B 参数 fp16 简化版 @latent 160²，稳态 0.264s，GPU peak 5.7 GB），跨帧 reshape/CrossFrameAttention/GroupEmbedding 三个自定义块逐一验证可实现，兜底路线 B 成立（本阶段未启用）。坑：GitHub 不可达→SwiftPM 依赖从本机缓存 vendor 化；metal 编译器需 DEVELOPER_DIR 指向 Xcode。脚本/报告入 git（`spikes/m1/`），mlpackage/npz/vendor 大二进制留本地不入库（见 .gitignore）。 |
| M2 静态化+小件 CoreML | ✅ | 2026-10-15 | embedding 包 `migration/m2/out/embeddings/`（15 MB 已入库）：24 tag×(prompt/pooled)+negative ''+Marigold 空 prompt[1,2,1024]，bf16@MPS 编码（与 M0 golden 同路径、调上游真实方法，逐 bit 一致）存 fp32，manifest 带 shape/dtype/md5+派生量(add_time_ids=[1280,1280,0,0,1280,1280],cfg=1.0/30步@1280,DDIM 4步@768)；纯 numpy 加载验收 51 tensor 全过。小件 4/4 转换+对齐达标：sdxl_dec 64.6dB/SSIM1.0、sdxl_enc dyn56.2dB、trans_dec 78.1dB/SSIM1.0、trans_enc dyn96.4dB（阈值 psnr01≥35/ssim≥0.99/dyn≥40），对齐基准为 Spike A 冻结 fp32 参考（输入源自 golden reconstruction/face.png）；CoreML fp16 前向 449/389/382/5.5ms。ANE 路径（CPU_AND_NE）逐模块加载+预测通过且指标仍过阈值。**compute plan 设备归属（M1 遗留）：4 模块计算算子 100% preferred=GPU、0 ANE**（405/305/1503/19 ops），与 UNet 一致，ANE 优化留后续 spike。产物：`migration/m2/`（scripts+README+results.json 入库；361 MB mlpackage 留本地）。坑：coremltools9 枚举是 CPU_AND_NE；compute_plan 需先 coremlcompiler 编译成 mlmodelc（Xcode 工具链绝对路径）；np.bool 需 bool() 转 JSON |
| M3 LayerDiff3D UNet | ✅ | 2026-10-15 | 合并单包 `migration/m3/out/models/layerdiff3d_unet_merged_fp16.mlpackage`（8.1 GB，EnumeratedShapes F∈{13,11}，替代 Spike B 两档共 15.2 GB；不入库）。group_index 消除：group embedding 两处作用对象均为静态量 ⇒ wrapper 置 `config.group_embedding=False` + 预计算条件落盘 `out/statics/`（eager 等价 134 dB）。轨迹锚点 `golden_trajectory/`（1.2 GB 不入库）：24/24 层与 M0 golden 逐 bit 一致。Swift 包 `migration/m3/swift/`（ldrunner，零依赖）：调度器 sigmas 与 diffusers 逐位一致；对齐：单步 UNet 67.6–68.2 dB、30 步轨迹末步 body 56.8/head 53.4 dB、decode 51.8–72.6 dB、**e2e vs golden min body 51.1/head 65.1 dB**（阈值 35，目检一致；误差主源 fp16 vs bf16 的 30 步混沌放大）。CoreML 耗时：稳态前向 body 5.4s/head 4.9s（Python bench），Swift 含封送 avg 7.4/6.1 s/step，30 步 222.9/183.9s；compute plan 12057 ops = 5621 GPU + 0 ANE（ANE 优化留独立 spike）。坑：repeat_interleave 符号 repeats → traced 图手术换 aten::expand；coremltools `_group_norm_impl` 符号 batch → 脚本内 mb.concat 补丁；HF hub 探测无代理挂死需 HF_HUB_OFFLINE=1；macOS 27 需 MLModel.compileModel；PNG 落盘是截断非四舍五入。详见 migration/m3/README.md |
| M4 Marigold UNet | ⬜ | | |
| M5 后处理+PSD writer | ⬜ | | |
| M6 SwiftUI app | ⬜ | | |

状态记号：⬜ 未开始 / 🚧 进行中 / ✅ 完成（验收通过）。
