# Stardew Valley NPC AI 对话系统

**角色**: 独立开发  |  **技术栈**: Python / FastAPI / LangChain / ChromaDB / LanceDB / BGE-M3

---

## 项目概述

为星露谷物语构建的 NPC AI 对话系统。C# 游戏客户端通过 HTTP 发送游戏状态 JSON，后端经三层记忆系统和 prompt 构建管道调用 LLM 生成 NPC 回复。支持 30+ 角色、多语言对话检索、原创角色扩展。

```
游戏客户端 → FastAPI → 状态机 → STM/MTM/LTM 记忆检索 → Few-shot 检索 → Prompt 组装 → LLM → 回复
```

---

## 核心技术

### 三层记忆系统

| 层级 | 存储 | 内容 |
|------|------|------|
| 短期 STM | JSON 文件 | 当前对话历史 |
| 中期 MTM | JSON 文件 | 当日对话摘要，LLM 提取候选事件/信念 |
| 长期 LTM | ChromaDB（1024 维 BGE-M3） | 5 个 collection：persona_seed / episodic_event / preference_belief / relationship_impression / narrative_arc |

- **Memory Recall 80%**（从 51% 提升）：经 query 重构 + 混合检索（向量 + topic_tag 关键词加分）+ ranker 场景标签加分 + BGE-M3 替换 all-MiniLM
- **混合检索**：向量召回 top_k×2 候选，topic_tags 匹配关键词 +0.15 similarity 重排
- **记忆衰减 & 写回**：时间衰减 + 重要性强化 + dormant/archived 自动降级
- **防剧透**：关系阶段 gate filter，stranger 不可见 marriage_dialogue

### Few-shot 示例检索

- LanceDB + BGE-M3 向量索引，39,000+ 条多语言游戏对话
- 分层回退链：dynamic → mixed → static → empty
- 反剧透过滤：relationship_gate + required_flags + route 三重门禁
- **原创角色白名单机制**：character_type == "original" 跳过 LanceDB，保护原创人设不被其他角色语料稀释

### Prompt 构建管道

- Jinja2 模板系统，5 段渲染（system → persona → memory → context → dialogue）
- 记忆使用引导指令，Memory Utilization 30% → 55%
- 上下文格式化：LTM 按 relative time 标注（昨天/3天前/几个月前）

### 日记结转

- END_DIALOGUE → LLM 提取 scene-level 中期候选（event + belief）
- END_DAY → 升级为长期记忆 + 信念更新 + 印象更新 + 衰减
- WEEKLY_REFLECTION → 第一人称内心独白 narrative_arc + relationship_delta

---

## 量化成果

| 指标 | 初始值 | 优化后 | 提升 |
|------|:---:|:---:|:---:|
| **Memory Recall** | 51% | **80%** | +29pp |
| **Layer Coverage** | 84% | **98%** | +14pp |
| **Contradiction Rate** | 1.1% | **0%** | ↓ |
| **Few-shot 胜率（Penny）** | — | **50% vs 11%** | +39pp |
| **Few-shot 胜率（Sebastian）** | — | **50% vs 22%** | +28pp |
| **Memory Utilization** | 30% | 55% | +25pp |

### 优化路径

```
all-MiniLM → query 重构 → 混合检索 → ranker tag bonus → BGE-M3 → 记忆引导指令
Recall: 51% → 69% → — → — → 80% → —
```

---

## 评估体系

- **90-case 盲测框架**：随机交换 A/B 顺序，judge LLM 不知哪侧有 few-shot
- **四维评估**：Retrieval Stats / Recall Correctness（Recall@K + Precision@K）/ Memory Use（utilization + contradiction + tone）/ A/B Comparison（winner + in_character + style + relevance）
- **105 条 LTM 测试数据**：5 NPC × 5 collection，第一人称对齐，确定性哈希 ID
- **30+ NPC citizen 认知网络**：每个 NPC 对戴蒙/宾馆项目的第一人称视角记忆

---

## 工程实践

- 自实现状态机（非 LangGraph），按 command 路由到 6 个 handler
- 临时 ChromaDB 隔离测试（tempfile.mkdtemp）
- Character type 白名单控制检索开关
- ChromaDB 维度不兼容检测（384 → 1024 迁移保护）
