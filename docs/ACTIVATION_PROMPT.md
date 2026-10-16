# 激活提示词（See-through macOS 原生迁移）

在新会话中粘贴以下全文即可激活本项目（当前阶段：M5）：

---

激活 See-through macOS 原生迁移 roadmap。

项目仓库在本机 ~/see-through（fork 自 shitagaki-lab/see-through，远端 Jup33Q/see-through）。
请先读固化 plan：~/see-through/docs/macos-native-roadmap.md，确认文末状态表中的当前阶段
（应为 M5 后处理 + PSD writer Swift 化），然后按该阶段的任务清单执行并跑该阶段的验收。

M3/M4 已完成（LayerDiff3D + Marigold 两个 UNet 与全部 VAE 小件 CoreML 化，Swift 调度器
/胶水/e2e 全过验收）。M5 是纯 CPU 后处理阶段，**不需要任何模型/CoreML 推理**，输入直接用
golden 冻结的中间产物（tests/golden/test_image/ 的 23 张 tag PNG + 23 张 *_depth.png +
src_img.png + info.json）。执行 M5 时的权威源码与可复用资产：

- 移植对象：`common/utils/inference_utils.py:436` 起的 `further_extr`（连通域、
  深度中值排序、可选左右拆分）→ `common/utils/io_utils.py:498-519` 的
  `dump_parts_psd`（psd-tools 写 PSD）。golden 运行参数：rotate=False、
  save_to_psd=True、tblr_split=False（LaMa inpainting 仅在 tblr_split 且有 hair 层时
  触发，默认路径不涉及，不迁移）。
- 对齐基准：tests/golden/test_image/test_image.psd + test_image.psd.json
  （M0 冻结；层名/层序/逐像素内容）。验收：Swift 产出 PSD 与 golden 结构/层名/顺序
  一致，用 psd-tools 读回 + 逐层像素对比双向验证。
- Swift 包 migration/m3/swift/（ldrunner）：NPY IO、PNG IO（RGBA 直通 alpha 语义已
  处理）、cv2 语义 resize、JSON 落盘框架均可复用；M5 子命令建议沿用 `m5-*` 或
  `psd-*` 前缀并落 JSON 到 migration/m5/out/（仿 M4 的 jsonWriteM4 模式）。
- 连通域/中值/排序：优先 Accelerate/vImage 或纯 Swift；语义以 scipy/上游实现为准
  （注意 M0 环境 scipy 已升 1.18.1）。
- 参考报告：migration/m3/README.md、migration/m4/README.md（对齐方法论与坑）。

环境与坑（M2–M4 实测，已验证的 workaround）：
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
  是独立 spike，不阻塞任何阶段验收。

工作规则：
1. 严格按 roadmap 的阶段顺序推进，不跳阶段。
2. 所有对齐以 tests/golden/ 下 M0 冻结的 golden 输出为基准（M5 为 golden PSD 与
   .psd.json）；轨迹级锚点 migration/m3、m4 的 golden_trajectory/ 留本地未入库，
   若缺失先重跑对应 01_dump_trajectory.py。
3. 优先 CoreML（ANE 优先）；设备/精度改动遵循 roadmap「技术事实底账」里列出的已知
   硬编码点，不要大范围重构上游代码。
4. 会话结束前：更新 roadmap 文末状态表（状态/日期/备注）与本文件（当前阶段），
   连同本阶段产物一起 commit 并 push 到 Jup33Q/see-through main，然后在回复中
   汇报本阶段验收结果。

---
