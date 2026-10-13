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

1. UNet trace/转换（固定 shape，body/head 两个 frame 档）→ fp16，必要时 palettize。
2. DPMSolverMultistep（sde-dpmsolver++，zero final sigmas）调度器 Swift 实现。
3. CFG 组装、源图 latent 编码、c_concat 条件拼装等管线胶水逻辑 Swift 化。
4. 逐阶段对齐：先对齐单步 UNet 输出，再对齐完整去噪轨迹，最后对齐解码 PNG。

验收：body/head 两 pass 输出 PNG 与 golden 视觉一致、逐层 PSNR 达标；记录 ANE 耗时/内存。

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
| M0 基线部署+golden | ⬜ | | |
| M1 可行性 spike + 决策门 | ⬜ | | 路线：未定 |
| M2 静态化+小件 CoreML | ⬜ | | |
| M3 LayerDiff3D UNet | ⬜ | | |
| M4 Marigold UNet | ⬜ | | |
| M5 后处理+PSD writer | ⬜ | | |
| M6 SwiftUI app | ⬜ | | |

状态记号：⬜ 未开始 / 🚧 进行中 / ✅ 完成（验收通过）。
