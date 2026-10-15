# 激活提示词（See-through macOS 原生迁移）

在新会话中粘贴以下全文即可激活本项目（当前阶段：M4）：

---

激活 See-through macOS 原生迁移 roadmap。

项目仓库在本机 ~/see-through（fork 自 shitagaki-lab/see-through，远端 Jup33Q/see-through）。
请先读固化 plan：~/see-through/docs/macos-native-roadmap.md，确认文末状态表中的当前阶段
（应为 M4 Marigold UNet CoreML），然后按该阶段的任务清单执行并跑该阶段的验收。

M3 已完成（LayerDiff3D UNet 合并单包 + Swift 调度器/胶水全过验收）。执行 M4 时可复用的
本机资产：

- UNet 转换链路：migration/m3/scripts/02_convert_unet_merged.py（含两个 coremltools 9.0
  符号形状补丁：traced 图 repeat_interleave→aten::expand 手术、_group_norm_impl 符号
  batch 的 mb.concat 补丁）。Marigold UNet 是同一个 UNetFrameConditionModel 类
  （24yearsold/seethroughv0.0.1_marigold），先查其 unet/config.json 是否有
  group_embedding——大概率没有，则无需求静态条件预计算，也无需 EnumeratedShapes
  （768px ⇒ latent 96²，单档固定 shape）。
- Swift 包 migration/m3/swift/（ldrunner：NPY IO、CoreML 封装、PNG IO、对齐 CLI 框架），
  M4 在其基础上加 DDIM 调度器（比 DPM 简单）与 marigold 子命令；调度器数学以本机
  diffusers 0.37.0 源码为唯一权威（DDIM：scheduling_ddim.py）。
- Marigold 空 prompt embedding：migration/m2/out/embeddings/marigold/（运行时无 CLIP）。
- golden：tests/golden/test_image/ 的 *_depth.png（逐层伪深度）+ info.json。
- 锚点 dump 参考实现：migration/m3/scripts/01_dump_trajectory.py（M4 需对
  apply_marigold 做同款轨迹 dump，存 migration/m4/golden_trajectory/）。
- 环境：conda env see_through（source /opt/miniconda3/etc/profile.d/conda.sh && conda
  activate see_through），coremltools 9.0 已装；HF 权重两 repo 均已在缓存
  （layerdiff3d snapshot 4477e6ce / marigold snapshot 4f4ffc46）。

注意（M2/M3 实测坑，已验证的 workaround）：
- 所有 from_pretrained 调用前必须 HF_HUB_OFFLINE=1（无代理时 HF hub 探测会挂死进程）；
  任何需要网络的操作用 HTTPS_PROXY=http://127.0.0.1:7890（git push 用
  git -c http.proxy=http://127.0.0.1:7890 push origin main）；hf 下载加
  HF_HUB_DISABLE_XET=1（xet CDN 不稳）。
- 跑推理前确认 ~/see-through/assets symlink 存在（ln -sf common/assets assets）。
- macOS 27 上 Swift 侧 MLModel(contentsOf:) 不再自动编译 .mlpackage，需先
  MLModel.compileModel(at:)（ldrunner 已处理）。
- golden 运行 cfg=1.0 ⇒ CFG 关闭，不要实现 CFG 双批/uncond 路径。
- computeUnits=ALL 下计算算子 100% 落 GPU（ANE 落盘率 0%），M4 验收记录 GPU 路径
  耗时/内存即可，ANE 优化是独立 spike。
- 上游 PNG 落盘是截断（astype(np.uint8)）而非四舍五入；apply_marigold 的深度后处理
  （normalize_depth=False 默认路径、*_depth.png 生成方式）以
  common/utils/inference_utils.py:192 起与 marigold_depth_pipeline.py 源码为准。

工作规则：
1. 严格按 roadmap 的阶段顺序推进，不跳阶段。
2. 所有模型对齐以 tests/golden/ 下 M0 冻结的 golden 输出为基准；轨迹级对齐以 M4 dump
   的 golden_trajectory 为基准。
3. 优先 CoreML（ANE 优先）；设备/精度改动遵循 roadmap「技术事实底账」里列出的已知
   硬编码点，不要大范围重构上游代码。
4. 会话结束前：更新 roadmap 文末状态表（状态/日期/备注），连同本阶段产物一起 commit
   并 push 到 Jup33Q/see-through main，然后在回复中汇报本阶段验收结果。

---
