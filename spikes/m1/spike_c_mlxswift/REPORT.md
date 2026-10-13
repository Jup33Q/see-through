# M1 Spike C：mlx-swift 兜底路线可行性报告

日期：2026-10-13 ｜ 机器：M5 Max / 128 GB / macOS 27 ｜ Swift 6.4 / Xcode toolchain（DEVELOPER_DIR 指向 Xcode）｜ mlx-swift main @ 0bb916c（本地 vendor）

## 结论：**兜底路线可行**

mlx-swift 跑通 SDXL 级 UNet 前向（随机权重 fp16，Metal GPU），项目特有的自定义块
（CrossFrame 跨帧 attention / Transformer3D 帧维 reshape / GroupEmbedding）在 MLX Swift 中
逐一验证可实现。作为「UNet 转不了 CoreML」时的路线 B 兜底成立。
（本次决策门 Spike B 已证明 UNet 能转 CoreML，故本 spike 仅作兜底证据存档。）

## 实测数据

bench：`swift run unetbench --iters 5`（debug 构建；同机无其他重负载时复测一次，日志
`logs/unetbench_run.log`）

| 项目 | 结果 |
|---|---|
| UNet 规模 | 1.99B 参数 fp16（SDXL 骨架 320/640/1280，简化版：无 13 帧维、attention 层数按 SDXL 标配） |
| 输入 | latent [1,160,160,4]（1280px 档）+ encoder_hidden [1,77,2048] + text_embeds + time_ids |
| warmup 前向（含 Metal kernel 编译） | 2.39 s |
| 稳态单次前向 | **0.264–0.275 s**（5 次均值 0.264 s） |
| GPU 内存 | active 3.99 GB / peak 5.70 GB |
| 输出正确性 | shape [1,160,160,4]，dtype fp16，全 finite |

注：本 bench 为简化 SDXL UNet（约真实 `UNetFrameConditionModel` 的一半参数量，且无 13/11 帧
批次维），用于验证「能跑 + 性能量级 + 自定义块可写」，不做数值对齐（随机权重）。

## 自定义块可实现性（`Sources/unetbench/CustomBlocks.swift`，全部通过）

- **跨帧 reshape**：`(B·F,S,C) ↔ (B·S,F,C)` permute 双射验证 maxErr=0.0，roundtrip==identity。
- **CrossFrameAttention**：帧维 attention [6,256,1280]→[6,256,1280] 前向通过（F=3 帧混合）。
- **GroupEmbedding**：`params[n_cls,dim]` 广播加 + Linear，3D/2D 两种输入形态均通过
  （对照 `layerdiff3d.py:533` 与 `transformer3d.py` 的语义）。

结论：三者都是 reshape/broadcast/attention/linear 的组合，MLX Swift 无任何实现障碍。

## 踩坑与对策

1. **GitHub 直连不可达**（M0 已记录）：SwiftPM 拉取 mlx-swift 及其传递依赖
   （swift-numerics / swift-argument-parser）失败。对策：mlx-swift 用本地 vendor 副本；
   两个传递依赖从本机 SwiftPM 缓存（`~/Library/Caches/org.swift.swiftpm/repositories/`）
   clone 到 `vendor/` 并把 `vendor/mlx-swift/Package.swift` 的 url 依赖改成本地 path。
2. **`metal` 编译器找不到**：`xcrun` 指向 CommandLineTools，CLT SDK 无 Metal 工具链。
   对策：`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build`
   （与 Spike A 的 coremlcompiler 坑同源）。
3. Swift 6 严格并发：main.swift 顶层代码隐式 @MainActor，全局 `forward()` 需标 `@MainActor`；
   `Module` 子类的无参 `init()` 需写 `override`。

## 产物

- SwiftPM 包：`Package.swift`、`Sources/unetbench/{main,UNet,CustomBlocks}.swift`
- 依赖：`vendor/mlx-swift`（+ 本地化的 swift-numerics / swift-argument-parser），均不入 git
- 运行日志：`logs/unetbench_run.log`

## 复现

```bash
cd spikes/m1/spike_c_mlxswift
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift run unetbench --iters 5
```
