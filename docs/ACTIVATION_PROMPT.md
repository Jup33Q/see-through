# 激活提示词（See-through macOS 原生迁移）

在新会话中粘贴以下全文即可激活本项目：

---

激活 See-through macOS 原生迁移 roadmap。

项目仓库在本机 `~/see-through`（fork 自 shitagaki-lab/see-through，远端
`Jup33Q/see-through`）。请先读固化 plan：`~/see-through/docs/macos-native-roadmap.md`，
确认文末状态表中的当前阶段，然后按该阶段的任务清单执行并跑该阶段的验收。

工作规则：
1. 严格按 roadmap 的阶段顺序推进，不跳阶段；M1 的决策门结论必须基于 spike 实测证据。
2. 所有模型对齐以 `tests/golden/` 下 M0 冻结的 golden 输出为基准（M0 未完成则先做 M0）。
3. 优先 CoreML（ANE）；UNet 若 CoreML 走不通按决策门退 mlx-swift（纯 Swift），
   Python 桥接是最后备选且需在状态表记录原因。
4. 设备/精度改动遵循 roadmap「技术事实底账」里列出的已知硬编码点，不要大范围重构上游代码。
5. 会话结束前：更新 roadmap 文末状态表（状态/日期/备注），连同本阶段产物一起
   commit 并 push 到 `Jup33Q/see-through` main，然后在回复中汇报本阶段验收结果。

---
