# 测试数据 — 人工审核

> 生成时间: 2026-06-12
> 对应 plan.md: Few-shot Quality Evaluation（第 1 步：Useful@3 人工评估）

---

## 文件说明

| 文件 | 内容 | 格式 |
|------|------|------|
| `ltm_data.json` | 5 个 NPC 的长期记忆数据，按 NPC → collection 组织 | JSON |
| `test_cases.jsonl` | 90 条对话测试用例 | JSONL（每行一条） |
| `generate_test_data.py` | 生成脚本（可重新生成） | Python |

---

## 数据规模

| 指标 | 数值 |
|------|------|
| NPC 数量 | 5（Damon, Abigail, Shane, Sebastian, Penny） |
| LTM 记忆总数 | **105 条** |
| 测试用例总数 | **90 条**（每 NPC 18 条） |

> **注**：只有 Damon（原创角色）有 `persona_seed`，和实际代码一致（`persona_seed.py` 仅实现了 `build_damon_persona_seed()`）。
> Abigail/Shane/Sebastian/Penny 是原版角色，角色信息来自 NPC JSON 配置 + LanceDB 对话数据。

### 各 NPC LTM 分布

| NPC | persona_seed | episodic_event | preference_belief | relationship_impression | narrative_arc | 小计 |
|-----|-------------|----------------|-------------------|------------------------|---------------|------|
| Damon | 6 | 12 | 4 | 1 | 3 | **26** |
| Abigail | — | 13 | 4 | 1 | 3 | **21** |
| Shane | — | 12 | 3 | 1 | 3 | **19** |
| Sebastian | — | 12 | 4 | 1 | 3 | **20** |
| Penny | — | 12 | 3 | 1 | 3 | **19** |

### 各 NPC 测试用例覆盖的关系阶段

| NPC | 覆盖的阶段（每阶段条数） |
|-----|-----------|
| Damon | stranger(3), acquaintance(3), friend(4), close friend(3), best friend(4) |
| Abigail | stranger(2), acquaintance(3), friend(3), close friend(2), best friend(1), dating(2), spouse(3) |
| Shane | stranger(2), acquaintance(3), friend(3), close friend(3), dating(2), spouse(3) |
| Sebastian | stranger(2), acquaintance(3), friend(3), close friend(2), best friend(1), dating(2), spouse(3) |
| Penny | stranger(2), acquaintance(3), friend(3), close friend(2), best friend(1), dating(2), spouse(3) |

### 各 NPC 测试用例覆盖的季节/天气/地点

| NPC | 季节覆盖 | 天气覆盖 | 地点覆盖 | 特殊标记 |
|-----|---------|---------|---------|---------|
| Damon | spring×4, summer×4, fall×5, winter×5 | Sun×9, Rain×9 | Town×5, Beach×3, Library×3, Mountain×2, Saloon×3 | is_birthday=1, is_festival=1, is_gifting=2, 短输入=1 |
| Abigail | spring×3, summer×4, fall×3, winter×4 | Sun×8, Rain×5, … | Town×5, Mountain×5, Farm×3, Beach×2, Library×1 | is_gifting=1 |
| Shane | spring×3, summer×3, fall×3, winter×3 | Sun×8, Rain×5, … | Town×3, JojaMart×1, Saloon×4, Forest×4, Farm×3 | is_gifting=2 |
| Sebastian | spring×4, summer×3, fall×3, winter×3 | Sun×8, Rain×5, … | Mountain×5, Farm×4, Saloon×2, Beach×3 | game_flags=3 |
| Penny | spring×3, summer×3, fall×3, winter×3 | Sun×8, Rain×5, … | Town×4, Library×3, Farm×4, Forest×2, Beach×1 | game_flags=3 |

---

## 审核要点

### 1. 角色一致性

每条 LTM 记忆的事件/信念是否与 NPC 官方设定一致？

- **Damon**：原创角色，人设见 `prompt_construction/npc/Damon.json`
- **Abigail/Shane/Sebastian/Penny**：原版角色，人设见对应 NPC JSON

### 2. 防剧透

- stranger 阶段的测试用例，LTM 中是否包含了只有高好感度才能知道的信息？
- 如果 NPC 是 stranger，检索到的 episodic_event 是否应该只包含 first_meeting 相关的内容？
- preference_belief 的 polarity 在低好感阶段是否合理？

### 3. 时序一致性

- episodic_event 的 `time`（游戏天数）是否递增合理？
- narrative_arc 的 `week_range` 是否与事件时间匹配？
- relationship_impression 的数值（trust/warmth/familiarity）是否与关系阶段匹配？

### 4. 测试用例覆盖

每个 NPC 的 18 条 case 是否覆盖了：
- [x] 不同季节（spring/summer/fall/winter）
- [x] 不同天气（Sun/Rain）
- [x] 不同地点（Town, Beach, Library, Mountain, Saloon, JojaMart, Forest, Farm）
- [x] 不同类型的 player_input（问候/提问/送礼/情感分享/短输入）
- [x] `is_gifting` 标记是否正确
- [x] 已婚 NPC 的 `game_flags` 是否包含 `relationship.married_to.XXX`
- [x] `is_birthday` / `is_festival` 边界情况
- [x] 极短输入边界情况（DAMON_11: "嗨。"）

### 5. expected_ltm_context

每个 case 的 `expected_ltm_context` 字段标注了人工期望检索到的记忆。审核时请确认：
- 这些记忆是否真的与该对话场景相关？
- 是否有遗漏的重要记忆未标注？

### 6. 边界情况

- [x] DAMON_11 的 `player_input` 是极短输入（"嗨。")
- 是否有 case 的场景与 NPC 的日常行程冲突？（如 Shane 白天在 Saloon 喝酒但他在 JojaMart 上班）
- 是否有 case 的 `today_actions` 与 `player_input` 矛盾？

---

## 如何审核

### 快速审核（推荐先做）

1. 用 VS Code / 任意编辑器打开 `ltm_data.json`
2. 折叠到 NPC 级别，逐个 NPC 展开 `episodic_event`
3. 通读事件内容，看是否符合角色设定
4. 打开 `test_cases.jsonl`，扫一遍 90 条 case 的 `player_input` 是否合理

### 详细审核

1. 选一个 NPC，对照其 `prompt_construction/npc/<NPC>.json` 中的 relationship_map
2. 逐条检查 LTM 中的 episodic_event 是否与 NPC 的 example 对话风格一致
3. 检查 test case 的 game_state 是否与 LTM 中的 memory 能匹配上

### 审核记录方式

在下方表格中标记每条 case 的审核结果：

| Case ID | 角色一致性 | 防剧透 | 场景合理 | 备注 |
|---------|-----------|--------|---------|------|
| DAMON_01 | ⬜ | ⬜ | ⬜ | |
| DAMON_02 | ⬜ | ⬜ | ⬜ | |
| ... | | | | |

---

## 使用方式

```bash
# 重新生成（修改了生成逻辑后）
python test_data/generate_test_data.py

# 将 LTM 数据导入 ChromaDB（待实现）
python test_data/load_ltm_to_chromadb.py --input test_data/ltm_data.json

# 跑 few-shot 评估
python test_data/run_few_shot_eval.py --cases test_data/test_cases.jsonl
```