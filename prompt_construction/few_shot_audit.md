# Few-Shot 系统评估报告

> 基于对 `plan.md` 的逐项对照 + 实际 LanceDB 数据审计 + 代码逻辑检查
> 生成时间: 2026-06-10
> 更新时间: 2026-06-10（static examples 补充 + 嵌入模型复用修复）

---

## 〇、修复记录

### 修复 1：嵌入模型复用（2026-06-10）

**问题**：`query_lancedb.py` 每次调用 `search_dialogues()` 都重新加载 `SentenceTransformer("BAAI/bge-m3")`，HuggingFace 429/SSL 错误时全链路 fallback。

**修复**：
- 新增模块级模型缓存单例 `get_embedding_model()`，线程安全
- `_resolve_hf_cache_path()` 自动解析本地 HF 缓存路径
- 默认 `TRANSFORMERS_OFFLINE=1`，首次失败自动重试在线
- `few_shot_provider.py` 检索前先确认模型就绪，debug 记录 `model_loaded`
- `graph.py` 启动时预热 BGE-M3
- `index_config.yaml` 新增 `embedding.local_path` 配置项

**验收**：首次加载 35s，后续 0.003s；`TRANSFORMERS_OFFLINE=1` 断网测试通过。

### 修复 2：static examples 均衡化（2026-06-10）

**问题**：15 个 NPC（除 Abigail）每阶段只有 1 条 static example（单字符串），fallback 质量极不均。

**修复**：
- 15 个 NPC 全部从 `example`（单字符串）→ `examples`（`{tag, content}` 列表）
- Damon：5-6 条/阶段（原创角色，最高优先级）
- 11 个可结婚 NPC：3-5 条/阶段
- 3 个非结婚 NPC：3 条/阶段
- 所有 examples 统一 `{"tag": "...", "content": "..."}` 格式
- tag 值包括：`low_heart`/`mid_heart`/`high_heart`、`rainy`/`sunny`、季节、地点等

**数据来源**：
- Sebastian / Haley / Emily / Sam — 来自星露谷物语中文 Wiki 对话
- 其他原版 NPC — 基于游戏原始对话 + 场景补充
- Damon — 原创对话，基于角色设定

**总量**：16 NPC，111 个关系阶段，449 条 examples（含 Abigail 63 条）

---

## 一、LanceDB 数据层现状

| 指标 | 实际值 | 计划目标 | 评估 |
|------|--------|----------|------|
| 总行数 | 39,204 | N/A | 数据量充足 |
| 角色数 | 39 个角色 | N/A | 覆盖全角色 |
| 语言数 | 12 种 (en/zh/de/es/fr/hu/it/ja/ko/pt/ru/tr) | 计划要求多语言 | **达标** |
| 对话类型 | general_dialogue: 27,636 / marriage_dialogue: 11,568 | 计划要求两者兼有 | **达标** |
| Damon 记录 | **0 条** | — | **严重缺失** |
| route 分布 | 全部为 "any" | 计划要求 route 分组 | **未实现 route 区分** |
| required_flags | 部分记录有婚姻标志 | 计划要求婚姻+岛屿标志 | 部分达标 |
| `all_dialogues.json` | **不存在**（按角色拆分） | 计划要求合并文件 | **格式偏差** |

### 1.1 Damon 问题

**发现**: LanceDB 中 Damon 有 0 条记录。这意味着：

- Damon 的 few-shot **永远** fallback 到 static
- 当前 `filter_by_relationship_gate()` 对 Damon 无意义（因为 LanceDB 根本没有 Damon 数据）
- Damon 的对话风格参考完全依赖 `npc/Damon.json` 中每个关系阶段的 1 条 `example`

**影响**:
- Damon 只有 1 条静态 example，作为 few-shot 风格参考**严重不足**
- 其他原创角色如果未来加入，也会有同样问题

**建议**: 需要为 Damon（及未来原创角色）构建 LanceDB 数据。可选方案：
1. 人工编写原创角色对话语料，导入 LanceDB
2. 用 LLM 批量生成符合角色设定的对话样本，人工审核后入库
3. 从实际游戏运行中积累高质量 Damon 对话，自动入库

### 1.2 route 全部为 "any"

**发现**: 所有 39,204 条记录的 `route` 字段都是 `"any"`。

**影响**:
- `filters.py` 中的 `route_visible()` 检查永远不会过滤任何记录
- 社区中心 vs Joja 路线对话区分**未实现**
- 虽然代码逻辑正确，但数据层缺少路线标记

**建议**: 在 canonical 提取阶段增加 route 判定逻辑（基于 dialogue_key 或文件来源），或在 processed 构建阶段手动标注。

### 1.3 `all_dialogues.json` 缺失

**发现**: 计划要求 `data/canonical/dialogue/all_dialogues.json`，实际数据按角色拆分为 39 个 `*_canonical.json`。

**影响**: 不影响功能（`build_processed_dialogues.py` 和 `build_lancedb_index.py` 能正确读取拆分文件），但与计划 deliverable D3 不符。

**建议**: 保持现状或生成一个合并文件。优先级低。

---

## 二、plan.md 逐项对照

### Phase 6: Retrieval query and fallback — 评估

| 计划要求 | 实现状态 | 详情 |
|----------|----------|------|
| **6.1 Target-language retrieval** — zh 查询优先返回 zh 记录 | **已实现** | `filters.filter_by_lang_fallback()` 实现 target→en→any 三级回退 |
| **6.2 English fallback** — zh 不足时回退 en | **已实现** | `DEFAULT_MIN_RESULTS=3`，低于此触发 en fallback，标记 `fallback_reason: "fallback_en"` |
| **6.3 Any-language fallback** — en 也不足时回退任意语言 | **已实现** | 标记 `fallback_reason: "fallback_any"` |
| **6.4 Game-state gate filtering** — required_flags + route 兼容 | **部分实现** | required_flags 过滤已实现；route 过滤代码存在但因数据全为 "any" 无实际效果 |

### Section 5.4 Retrieval tests — 评估

| 计划要求的测试用例 | 实现状态 |
|-------------------|----------|
| zh 查询返回 zh 记录优先 | `test_query_filters.py` 覆盖 |
| en 查询返回 en 记录优先 | `test_query_filters.py` 覆盖 |
| 无 married flag 时婚姻对话隐藏 | `test_query_filters.py` 覆盖 |
| 有 married flag 时婚姻对话可见 | `test_query_filters.py` 覆盖 |
| 无 resort flag 时岛屿对话隐藏 | `test_query_filters.py` 覆盖 |
| en fallback 行为确定性可见 | `test_query_filters.py` 覆盖 |

### 额外实现的测试（超出计划）

| 测试 | 文件 |
|------|------|
| Rich query 构造（不含长期记忆） | `test_few_shot_provider.py` |
| 关系阶段防剧透过滤 | `test_few_shot_provider.py` |
| Heart level 推导 | `test_few_shot_provider.py` |
| 分层 fallback（dynamic/mixed/static/empty） | `test_few_shot_provider.py` |
| FewShotResult 结构化返回 | `test_few_shot_provider.py` |
| format 输出不暴露内部信息 | `test_few_shot_provider.py` |
| 模板 empty 时不渲染 few-shot section | `test_few_shot_provider.py` |

---

## 三、few-shot 管线质量评估

### 3.1 已完成且正确的部分

| 项目 | 状态 | 说明 |
|------|------|------|
| 分层 fallback | **正确** | dynamic→mixed→static→empty，4 级降级 |
| 防剧透过滤 | **正确** | stranger 看不到 married/heart_min_4 等高级内容 |
| Rich query | **正确** | 含 NPC+输入+关系+地点+天气+态度，不含长期记忆 |
| 格式化不泄露内部信息 | **正确** | score/example_id/metadata 不暴露给模型 |
| 模板条件渲染 | **正确** | empty 时不渲染 few-shot section |
| Debug 日志 | **正确** | 打印 source/count/fallback_reason，不打印完整 prompt |
| 静态 example 格式统一 | **正确** | `- [tag] content`，tag 为空时省略 |

### 3.2 存在风险/需改进的部分

#### 风险 1: 嵌入模型在线加载依赖

**现象**: 当前 `query_lancedb.py` 每次调用都重新加载 `SentenceTransformer("BAAI/bge-m3")`，且依赖 HuggingFace/镜像在线下载模型。

**审计证据**: 测试运行时频繁出现 `429 Too Many Requests` 或 SSL 错误。

**影响**: 如果 HuggingFace 不可达，LanceDB 检索**完全不可用**，所有 NPC 全部 fallback 到 static。

**建议**:
1. 在 `graph.py` 初始化阶段预加载模型（`graph.py` 已经有 `_ = store.embedding_function` 预热，但 `query_lancedb.py` 没有复用）
2. 将模型缓存到本地路径，避免在线下载
3. 考虑让 `few_shot_provider` 接受预初始化的嵌入函数，而非每次新建

#### 风险 2: Damon 及原创角色 zero-shot 问题

**现象**: Damon 在 LanceDB 中 0 条记录，每个关系阶段仅 1 条静态 example。

**审计证据**: LanceDB `character="Damon"` → 0 rows；`npc/Damon.json` 每阶段仅 1 条 `example`。

**影响**: Damon 的 few-shot 只有 1 条参考，LLM 缺乏足够风格样本，回复质量可能不稳定。

**建议**: 见 1.1 节。

#### 风险 3: 静态 examples 质量不均

**现象**:
- Abigail 每阶段 9-14 条 examples（含 tag 标签）
- Damon 每阶段仅 1 条（无 tag）
- Sebastian/Haley/Emily 等每阶段也仅 1 条

**审计证据**: 静态 examples 审计输出。

**影响**: Abigail 的静态 fallback 远比其他角色丰富，导致 fallback 质量不一致。

**建议**: 为其他角色补充更多关系阶段的 examples（可从 LanceDB 提取高质量 zh 记录作为静态 examples）。

#### 风险 4: `_relationship_to_stage_value` 映射不完整

**现象**: `_STAGE_VALUES` 只覆盖了常见关系字符串。游戏中可能出现 `heart_min_2`、`heart_min_3` 等非常规值。

**当前处理**: 未匹配时返回 -1（不过滤），这是一个安全默认但可能导致不该看到的对话漏过。

**审计证据**: `filter_by_relationship_gate` 测试中 `unknown_relationship_passes_all` 用例通过。

**建议**: 对于 `_relationship_to_stage_value` 返回 -1 的情况，记录 warning 日志，方便排查。

#### 风险 5: `few_shot_provider` 与 `graph.py` 初始化模型的复用

**现象**: `graph.py` 在启动时预加载了 `MemoryStore` 的嵌入函数，但 `few_shot_provider` → `query_lancedb` 每次都新建 `SentenceTransformer`。

**影响**: 重复加载模型 → 内存浪费 + 首次调用延迟 + HuggingFace 请求频率过高。

**建议**: 让 `query_lancedb` 支持传入预加载的模型实例，或在 `few_shot_provider` 层缓存模型。

---

## 四、plan.md "Next Milestone" 对照

plan.md 第 9 节列出了下一里程碑的候选方向：

| 方向 | 当前状态 | 是否在本次范围内 |
|------|----------|-----------------|
| LLM semantic annotation | 未实现 | 否 |
| topic_tags / intent / tone | 未实现 | 否 |
| validation reports | 未实现 | 否 |
| review_queue | 未实现 | 否 |
| Festivals extraction | 未实现 | 否 |
| Strings whitelist extraction | 未实现 | 否 |
| MMR diversity selector | 未实现（当前用 scene diversity 代替） | 否 |
| prompt assembler | **已实现** (prompt_builder + context_formatter + template) | 已完成 |

---

## 五、总结评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 计划 Phase 6 覆盖度 | **9/10** | route 数据缺失扣 1 分 |
| 测试覆盖度 | **9/10** | 50 个 few-shot 测试 + query_filters 测试充分；缺 LanceDB 集成测试（因在线依赖） |
| 防剧透安全性 | **10/10** | gate filter 11 个用例全部通过，数值映射合理 |
| Fallback 健壮性 | **8/10** | 4 级降级逻辑正确；但 HuggingFace 不可达时全部降为 static，Damon 只有 1 条 example |
| 代码可维护性 | **9/10** | 职责清晰：provider → selector → formatter → template |
| 数据完整性 | **6/10** | Damon=0 条、route 全 "any"、静态 examples 严重不均 |

### 最优先待办

1. **为 Damon 建语料入库** — 否则原创角色 few-shot 形同虚设
2. **复用嵌入模型** — 避免每次检索重新加载模型
3. **补充其他 NPC 静态 examples** — 至少每个关系阶段 3 条
4. **route 标注** — 让 route 过滤实际生效
