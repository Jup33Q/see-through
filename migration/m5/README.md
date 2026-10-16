# M5 — 后处理与 PSD 导出 Swift 化报告

日期：2026-10-16 ｜ 机器：M5 Max / 128 GB / macOS 27 ｜ Swift 侧并入 M3 包 `migration/m3/swift/`（ldrunner 新增 `m5-post`）｜ 本阶段无模型或 CoreML 推理

范围：`further_extr` 的 M0 默认路径（`rotate=False`、`save_to_psd=True`、`tblr_split=False`）以及可选连通域左右拆分；原生 8-bit RGBA PSD writer、RLE、深度中值稳定排序和 `.psd.json` 旁车。LaMa hair inpainting 与可选 `*_depth.psd` 不在本次产物内。

## 实现

- `PostProcessing.swift`：按 `info.json` 原始顺序加载 23 个槽位并过滤空部件；alpha>10 mask、两次 boundingRect 裁剪、float32 深度中值、nose/mouth 原图 RGB 回填、face 关联深度修正、稳定降序排序。
- 纯 Swift 8 邻接连通域与 stats、eyes 四组件拆分、handwear/eye tags/ears 可选左右拆分、3×3 cross dilation。`process_cuts` 保留上游 `1-(1-depth)*mask` 的浮点运算顺序，避免数学化简造成 1 ulp 中值偏差。
- `PSDWriter.swift`：零依赖 PSD v1 二进制 writer；层通道顺序 `-1/0/1/2/-2`，与 psd-tools 1.14.2 的 RGBA 输入行为一致（alpha 同时写 transparency 与 user mask）；所有层通道和 merged preview 均用 PackBits RLE。
- `m5-post` 默认直接读取 `tests/golden/test_image/`，输出 `migration/m5/out/psd/test_image.psd` 与 `.psd.json`；报告 JSON 按 M4 模式落到 `migration/m5/out/`。
- `scripts/01_make_split_fixture.py` 生成 golden 不会触发的 eyes/handwear 拆分上游 oracle；`scripts/02_validate_psd.py` 用 psd-tools 读回 Swift PSD，逐层检查结构、通道、RLE、RGBA/mask 像素、旁车 JSON，并做 psd-tools 再保存 roundtrip。

## 验收结果

| 对齐项 | 结果 |
|---|---|
| M0 golden：文档/17 层名称/顺序/bbox/通道/RLE | **PASS** |
| M0 golden：17/17 层 RGBA + alpha mask | **逐像素完全一致** |
| M0 golden `.psd.json` | **完全一致** |
| psd-tools 读回 → 再保存 → 再读回 | **PASS（17/17）** |
| 合成 eyes + handwear 左右拆分 fixture | **PASS（6/6 层、mask、JSON、roundtrip 全一致）** |
| macOS `sips` 独立读取 | **PASS（1280×1280，hasAlpha=yes）** |
| M3 DPM / M4 DDIM 无模型 fixture 回归 | **PASS / PASS** |

Swift PSD 为 2,469,938 bytes，`m5-post` 本次总耗时 318.7 ms（后处理 289.0 ms，其余为 PSD/RLE 写盘）。完整指标在 `out/psd_alignment.json`、`out/split_alignment.json`、`out/m5_post.json` 和 `out/split_fixture_report.json`。

辅助项：merged preview 与 golden 有 9,811 个 RGB 值相差 1/255，不影响 17 层逐像素验收；合成 fixture 的 merged preview 完全一致。本机未安装 Photoshop，因此未声称 Photoshop 实机验证；第二读取路径由 psd-tools roundtrip 与 macOS `sips` 完成。

## 复现

```bash
cd migration/m3/swift
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
swift build -c release
cd ../../..

# M0 golden 默认路径
migration/m3/swift/.build/release/ldrunner m5-post --out m5_post.json
/opt/miniconda3/envs/see_through/bin/python migration/m5/scripts/02_validate_psd.py \
  --actual migration/m5/out/psd/test_image.psd \
  --reference tests/golden/test_image/test_image.psd \
  --actual-json migration/m5/out/psd/test_image.psd.json \
  --reference-json tests/golden/test_image/test_image.psd.json \
  --report migration/m5/out/psd_alignment.json

# 连通域与可选左右拆分
HF_HUB_OFFLINE=1 /opt/miniconda3/envs/see_through/bin/python \
  migration/m5/scripts/01_make_split_fixture.py
migration/m3/swift/.build/release/ldrunner m5-post \
  --src migration/m5/out/split_fixture/input \
  --psd migration/m5/out/split_fixture/swift.psd \
  --tblr-split true --out split_fixture_report.json
/opt/miniconda3/envs/see_through/bin/python migration/m5/scripts/02_validate_psd.py \
  --actual migration/m5/out/split_fixture/swift.psd \
  --reference migration/m5/out/split_fixture/input.psd \
  --actual-json migration/m5/out/split_fixture/swift.psd.json \
  --reference-json migration/m5/out/split_fixture/input.psd.json \
  --report migration/m5/out/split_alignment.json
```

生成的 PSD、fixture 和上游 scratch 均留本地（见 `.gitignore`）；脚本与 JSON 验收报告入库。
