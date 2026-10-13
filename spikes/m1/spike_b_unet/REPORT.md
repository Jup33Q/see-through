# M1 Spike B — UNetFrameConditionModel CoreML 转换可行性

日期：2026-10-13。机器：M5 Max / 128 GB / macOS 27（**同机有其他 spike 任务并行，耗时数据偏悲观**）。

## 对象

- 模型：`layerdifforg/seethroughv0.0.2_layerdiff3d` UNet 子目录，`UNetFrameConditionModel`（SDXL 骨架 320/640/1280，`use_linear_projection=True`，4.07B 参数 fp32）。
- 目标固定 shape：1280px → latent 160×160；body 档 `(1, 13, 8, 160, 160)`（latent4 + c_concat4），head 档 `(1, 11, 8, 160, 160)`；`encoder_hidden_states (F,77,2048)`、`text_embeds (F,1280)`、`time_ids (F,6)`；`group_index` body=0 / head=1。
- 转换链：`torch.jit.trace`（CPU fp32）→ `coremltools.convert`（mlprogram, fp16 compute_precision, `compute_units=ALL`, `minimum_deployment_target=macOS15`）。

## 环境

- python 3.12 / torch 2.14.1 (MPS) / diffusers 0.37.0 / transformers 5.0.0 / coremltools **9.0**（本 spike 新装）。
- ⚠️ coremltools 9.0 官方测试上限 torch 2.7.0；2.14.1 属未测试组合（目前未观察到因此导致的问题）。

## 检查点结论（随实验更新）

### 0. PyTorch 参考前向（MPS fp32）

| 档位 | shape | 单次前向 | 输出统计 |
|---|---|---|---|
| body 13 帧 | (1,13,8,160,160) | 19.3 s | mean -0.0051 / std 0.9998 |
| head 11 帧 | (1,11,8,160,160) | 16.2 s | mean -0.0057 / std 0.9976 |

产物：`ref_in_{tag}.npz`（固定种子 1234 输入）、`ref_out_{tag}.npy`、`ref_meta_{tag}.json`。

### 1. 最小复现探针（全部通过）

| 探针 | 内容 | trace | ct.convert+predict | 数值差 (fp16 vs fp32 ref) |
|---|---|---|---|---|
| `sdpa` | `F.scaled_dot_product_attention`（20头×64dim, seq 1024） | OK | OK | max_abs 6.8e-05 |
| `group_embedding` | `GroupEmbedding(13,2048)`，3D 输入 `x + params[:,None]` | OK | OK | max_abs 1.9e-03 |
| `transformer3d_midblock_attn0` | 真实权重 mid block 第 1 个 `Transformer3DModel`（13帧, 1280ch, 40×40, 10 层 spatial + 5 层 CrossFrame temporal） | OK (82s CPU) | OK (10s) | max_abs 7.2e-02, mean_abs 3.3e-03 |

结论：SDPA、frame reshape/permute、GroupEmbedding 三个风险点在 coremltools 9.0 下**单独均可转换**，SDPA 无需手工拆 einsum。

### 2. 整体 trace + 转换（body13 @ 160²）—— **全链路通过**

| 检查点 | 结果 | 耗时/数据 |
|---|---|---|
| `torch.jit.trace`（CPU fp32, strict=True） | **OK**，无 monkey-patch、无注意力重写 | 794.8 s；仅有 benign TracerWarning（shape assert / freeu 分支等常量折叠，固定 shape 下无影响） |
| traced vs eager 对齐 | max_abs **1.19e-05** / mean_abs 7.3e-07（max_rel 大仅出现在近零值） | traced 前向 158 s（CPU） |
| `ct.convert`（mlprogram, fp16, ALL, macOS15） | **OK** | 65.8 s（快于预期，4B 参数） |
| `.mlpackage` 落盘 | 7.6 GB | 加载 13.9 s |
| `compile_model` → `.mlmodelc` | **OK** | 0.5 s（编译落盘成功） |
| `predict`（compute_units=ALL） | **OK** | 首次 18.8 s（含编译），稳态 **5.1 / 5.2 / 5.3 s** |
| 数值对齐 vs PyTorch MPS fp32 参考 | max_abs **3.53e-03** / mean_abs 2.89e-04 / MSE 1.48e-07 | **PSNR 82.36 dB**（peak=5.04） |
| 峰值 RSS | 43.4 GB（trace+convert 同进程；推理时约 15 GB 量级） | — |

- SDPA → coremltools 9.0 直接映射（见探针 `sdpa`），**不需要** apple/ml-stable-diffusion 的 split-einsum 重写。
- frame reshape（`(B,F,C,H,W)→(B·F,C,H,W)` 与 Transformer3D 内 `(B·F,S,C)→(B·S,F,C)`）全部静态化通过。
- `GroupEmbedding`（body/head 靠 `group_index` 选 ModuleList 分支）在 wrapper 里固化为常量索引，两档需各转一份（权重共享、仅 group embedding 参数不同）。
- **ANE 落盘率**：`MLComputePlan` 统计显示全部 10380 个算子中，const 5626 个（无设备归属），其余 **4754 个算子 100% preferred=GPU，0 个 NeuralEngine**。即 compute_units=ALL 下 ANE 颗粒无收——fp16 SDPA/大 conv 未被编译器判给 ANE。若要上 ANE 需按 ml-stable-diffusion 手法拆 attention（split-einsum）+ 可能量化，留作后续实验。
- 性能对比：CoreML fp16（5.1 s）vs PyTorch MPS fp32（19.3 s），同机并行任务下约 **3.8×**。注意精度档位不同（fp16 vs fp32），且 CoreML 数字来自 ANE/GPU 编译后执行。

### 3. 整体 trace + 转换（head11 @ 160²，group_index=1）—— **同样通过**

| 检查点 | 结果 |
|---|---|
| trace | OK，664.3 s；trace-vs-eager max_abs 1.14e-05 |
| convert | OK，69.5 s；mlpackage 7.6 GB |
| predict | 首次 16.1 s，稳态 **4.2 / 4.3 / 4.4 s** |
| 数值对齐 | max_abs **3.22e-03** / mean_abs 2.88e-04 / **PSNR 82.26 dB** |

## 最终结论

**能转。** `UNetFrameConditionModel` 在 coremltools 9.0 + torch 2.14.1 下，`torch.jit.trace` → `ct.convert(fp16, ALL)` 全链路零修改通过（仅需一个固定 `group_index`、扁平张量入参的薄 wrapper），body13 / head11 两档均编译落盘、推理数值对齐（PSNR ≈ 82 dB，max_abs ≈ 3e-3，纯 fp16 舍入量级）。原先担心的三个风险点（frame reshape、Transformer3D/CrossFrame temporal attention、GroupEmbedding）经最小复现探针逐一验证均兼容；SDPA 被 coremltools 原生映射，无需 split-einsum。

**遗留/注意项**：
1. **ANE 利用率为 0**——compute_units=ALL 下所有计算算子落在 GPU（4754/4754）。若想压 ANE 功耗/让出 GPU，需按 apple/ml-stable-diffusion 手法拆 attention 并可能量化，作为后续独立 spike。
2. `group_index` 是 Python 分支（ModuleList 索引），两档各产一份 7.6 GB mlpackage；权重 99% 重复，后续可考虑把 group embedding 改为张量输入以合并成单模型（每组 embedding 仅 13×1280 + 13×2048 参数，改造量小）。
3. 固定 shape：160² latent / 13 / 11 帧写死；换分辨率需重转。5D 输入张量在 macOS15 target 下工作正常。
4. coremltools 9.0 未官方测试 torch 2.14.1；本次未观察到问题，但属非官方组合。
5. 磁盘：`traced_{tag}.pt` 各 16.3 GB（中间产物，可删）；`unet_{tag}.mlpackage` 各 7.6 GB。
6. 耗时为同机并行其他 spike 任务时的数据，偏悲观。

## 复现

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate see_through
cd spikes/m1/spike_b_unet
python ref_forward.py --frames 13 --group 0 --size 160 --device mps --tag body13_160
python min_repro.py all          # sdpa / group / t3d 三个探针
python convert_unet.py --frames 13 --group 0 --size 160 --tag body13_160
```

日志：`ref_*.log`、`convert_*.log`、`min_repro_results.log`、`convert_result_*.json`。
