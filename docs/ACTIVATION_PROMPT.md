# 激活提示词（See-through macOS 原生迁移）

在新会话中粘贴以下全文即可激活本项目（当前阶段：M6）：

---

激活 See-through macOS 原生迁移 roadmap。

项目仓库在本机 ~/see-through（fork 自 shitagaki-lab/see-through，远端 Jup33Q/see-through）。
请先读固化 plan：~/see-through/docs/macos-native-roadmap.md，确认文末状态表中的当前阶段
（应为 M6 SwiftUI App 整合），然后按该阶段的任务清单执行并跑该阶段的验收。

M3/M4/M5 已完成：LayerDiff3D 与 Marigold UNet、全部 VAE 小件已 CoreML 化，Swift
调度器/胶水/e2e 全过验收；纯 Swift 后处理与原生 RLE PSD writer 已对齐 M0 golden
（17/17 层逐像素一致）。M6 按 roadmap 整合 macOS SwiftUI app：拖图、分段进度、
分层预览、PSD 导出、模型包管理与 benchmark。可复用资产：

- Swift 包 migration/m3/swift/（ldrunner）：CoreML 封装、两套调度器与推理胶水、
  NPY/PNG/JSON IO；`e2e --pass body|head`、`mari-e2e`、`m5-post` 是已验收的调用参考。
- 原生后处理与 PSD：migration/m3/swift/Sources/ldrunner/PostProcessing.swift 和
  PSDWriter.swift；M5 报告与复现命令见 migration/m5/README.md。
- CoreML 模型（本地大二进制，不入库）：migration/m2/out/models/ 的 4 个小件、
  migration/m3/out/models/layerdiff3d_unet_merged_fp16.mlpackage、
  migration/m4/out/models/ 的 Marigold UNet/VAE。静态条件与 embedding 分别见
  migration/m3/out/statics/、migration/m2/out/embeddings/。
- 对齐基准：tests/golden/test_image/ 的最终 PNG、深度、PSD 与 .psd.json；
  分阶段数值和 M0 MPS 性能基线见 migration/m3/README.md、migration/m4/README.md、
  migration/m5/README.md 及 roadmap 状态表。

环境与坑（M2–M5 实测，已验证的 workaround）：
- 所有 from_pretrained 调用前必须 HF_HUB_OFFLINE=1（无代理时 HF hub 探测会挂死进程）；
  需要网络的操作用 HTTPS_PROXY=http://127.0.0.1:7890（git push 用
  git -c http.proxy=http://127.0.0.1:7890 push origin main）。
- 跑上游推理前确认 ~/see-through/assets symlink 存在（ln -sf common/assets assets）。
- macOS 27 上 Swift 侧 MLModel(contentsOf:) 需先 MLModel.compileModel(at:)（ldrunner
  已处理）；swift build 若报 metal 工具链错，export DEVELOPER_DIR=/Applications/
  Xcode.app/Contents/Developer。
- 上游 PNG 落盘是截断（astype(np.uint8)）而非四舍五入；cv2 INTER_AREA 8U 是定点
  实现（与浮点面积平均有 1/255 级差异，M4 已确认无害）。
- JSONSerialization 遇 ±inf 抛 NSException（ldrunner jsonWriteBase 已清洗）；
  M4 VAE decoder 输出是 fp16 MLMultiArray（toFloatArray 已支持）。
- computeUnits=ALL 下所有已转模型计算算子 100% 落 GPU（ANE 落盘率 0%），ANE 优化
  是独立 spike，不阻塞任何阶段验收；benchmark 应记录真实设备归属，不把 GPU 写成 ANE。

工作规则：
1. 严格按 roadmap 的阶段顺序推进，不跳阶段。
2. 所有对齐以 tests/golden/ 下 M0 冻结的 golden 输出为基准；轨迹级锚点
   migration/m3、m4 的 golden_trajectory/ 留本地未入库，若缺失先重跑对应
   01_dump_trajectory.py。
3. 优先 CoreML（ANE 优先）；设备/精度改动遵循 roadmap「技术事实底账」里列出的已知
   硬编码点，不要大范围重构上游代码。默认路径 cfg=1.0 无 CFG；LaMa 不在迁移范围。
4. 会话结束前：更新 roadmap 文末状态表（状态/日期/备注）与本文件（当前阶段），
   连同本阶段产物一起 commit 并 push 到 Jup33Q/see-through main，然后在回复中
   汇报本阶段验收结果。

---
