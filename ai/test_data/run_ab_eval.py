"""
A/B 评估主脚本：few-shot ON vs OFF + Memory Recall + Memory Use 质量检查。

用法:
    # 小规模测试（5 条）
    python test_data/run_ab_eval.py --limit 5 --output test_data/ab_eval_test.json

    # 全量运行
    python test_data/run_ab_eval.py --output test_data/ab_eval_report.json

    # 指定模型
    python test_data/run_ab_eval.py --model deepseek-chat --temperature 0.5 --limit 10
"""

import json
import sys
import time
import random
import argparse
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 可复现性工具
# ---------------------------------------------------------------------------

def _hash_file(filepath: Path) -> str:
    """SHA256 of a file's content, truncated to 16 hex chars."""
    try:
        return hashlib.sha256(filepath.read_text(encoding="utf-8").encode()).hexdigest()[:16]
    except Exception:
        return "MISSING"


def _hash_case_content(case: dict) -> str:
    """Content hash of a test case for drift detection.

    Hashes everything except case_id, so renumbering cases doesn't
    change the hash — only content changes do.
    """
    content = {k: v for k, v in case.items() if k != "case_id"}
    raw = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _compute_prompt_template_hash() -> str:
    """Combined SHA256 of prompt template source files.

    If any of these files change, the hash changes → prompts may differ.
    """
    template_files = [
        "prompt_construction/prompt/prompt_builder.py",
        "prompt_construction/prompt/context_formatter.py",
        "prompt_construction/prompt/prompt_damon.py",
        "prompt_construction/prompt/prompt_others.py",
    ]
    h = hashlib.sha256()
    for rel_path in template_files:
        fp = _PROJECT_ROOT / rel_path
        h.update(_hash_file(fp).encode())
    # Also hash this eval script itself (changes here affect results)
    h.update(_hash_file(Path(__file__)).encode())
    return h.hexdigest()[:16]


def _snapshot_config() -> dict:
    """Load config.json, redact sensitive fields, return key params."""
    try:
        cfg_path = _PROJECT_ROOT / "config.json"
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        safe = {}
        for k in ("Provider", "ModelName", "Temperature", "ServerAddress",
                   "MaxTokens", "TopP", "FrequencyPenalty", "PresencePenalty"):
            if k in raw:
                safe[k] = raw[k]
        # Redact API key presence
        if "ApiKey" in raw:
            key = raw["ApiKey"]
            safe["ApiKey"] = f"***{key[-4:]}" if key and len(key) > 4 else "***"
        return safe
    except Exception as e:
        return {"error": str(e)}
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievalStats:
    """检索数量统计"""
    persona_count: int = 0
    episodic_count: int = 0
    relation_count: int = 0
    total_retrieved: int = 0
    total_available: int = 0
    avg_similarity: float = 0.0
    layer_coverage: float = 0.0        # 3 层中有多少层有结果
    all_retrieved: List[Dict] = field(default_factory=list)


@dataclass
class RecallCorrectness:
    """召回正确性：基于 expected_memory_ids 计算"""
    expected_count: int = 0             # expected_memory_ids 总数
    matched_count: int = 0              # 实际检索到的匹配数
    retrieved_count: int = 0            # 总检索数
    recall: float = 0.0                 # matched / expected（0-1）
    precision: float = 0.0              # matched / retrieved（0-1）
    matched_ids: List[str] = field(default_factory=list)
    missed_ids: List[str] = field(default_factory=list)


@dataclass
class QualityVerdict:
    """单条回复质量评估（简化版：只测 OOC + 相关性 + 矛盾）"""
    character_fidelity: int = 1   # 0-3 角色还原度
    relevance: int = 1            # 0-3 回复相关性
    contradiction: int = 0        # 0-3 事实矛盾/幻觉（越低越好）


@dataclass
class ABVerdict:
    """A/B 对比 judge 评估结果"""
    winner: str = "tie"          # "A" | "B" | "tie"
    a_character_fidelity: int = 1
    b_character_fidelity: int = 1
    a_relevance: int = 1
    b_relevance: int = 1
    a_contradiction: int = 0
    b_contradiction: int = 0
    reason: str = ""


@dataclass
class CaseResult:
    """单条 case 的完整结果"""
    case_id: str = ""
    npc_id: str = ""
    relationship: str = ""
    player_input: str = ""

    # 响应（judge 看到的标签 A/B，可能是交换过的）
    response_a: str = ""
    response_b: str = ""
    error_a: str = ""
    error_b: str = ""

    # few-shot 信息
    few_shot_source: str = "empty"
    few_shot_count: int = 0

    # 盲测映射：True 表示 A=few-shot, B=no-few-shot；False 表示 A=no-few-shot, B=few-shot
    a_is_few_shot: bool = True

    # memory retrieval stats + recall correctness
    retrieval_stats: Optional[RetrievalStats] = None
    recall_correctness: Optional[RecallCorrectness] = None

    # 质量评估（judge 标签 A/B，后续需根据 a_is_few_shot 映射回真实 variant）
    quality_a: Optional[QualityVerdict] = None
    quality_b: Optional[QualityVerdict] = None

    # A/B 对比（judge 标签 A/B，后续需根据 a_is_few_shot 映射回真实 variant）
    ab_verdict: Optional[ABVerdict] = None

    # 完整 prompt（可追溯）
    prompt_fs_on: str = ""
    prompt_fs_off: str = ""

    # 分阶段耗时（ms）
    timing_ms: Dict[str, float] = field(default_factory=dict)

    # 总耗时
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0
    latency_judge_ms: float = 0.0


# ---------------------------------------------------------------------------
# 核心管道
# ---------------------------------------------------------------------------

class ABEvalRunner:
    def __init__(self, store, model_name: str = "", temperature: float | None = None):
        """
        Args:
            store: MemoryStore 实例（已导入测试数据）
            model_name: 模型名，空字符串表示用 config.json 的配置
            temperature: 温度，None 表示用 config.json 的配置
        """
        self.store = store
        self.model_name = model_name
        self.temperature = temperature

        # 初始化各个子模块（延迟导入，避免启动时大量加载）
        from memory.LTMemory import MemoryRetriever, MemoryRanker, ContextManager
        from llm import create_llm

        self.retriever = MemoryRetriever(store)
        self.ranker = MemoryRanker()
        self.ctx_manager = ContextManager(self.retriever, self.ranker)
        self.llm = create_llm(
            model_name=model_name or None,
            temperature=temperature,
        )

    # ---- Memory Retrieval ----

    def retrieve_memories(self, state: dict, npc_config: dict,
                          expected_memory_ids: List[str] = None) -> tuple:
        """检索记忆，返回 (formatted_dict, raw_results, retrieval_stats, recall_correctness)"""
        rel_desc = npc_config.get("description", "")
        rel_instr = npc_config.get("instruction", "")
        player_input = state.get("last_user_input", "")

        # P1 fix: 截断过长 relationship description，避免稀释 player_input 的向量信号。
        # stranger 阶段的 desc 可达 60+ 字（如"完全的陌生人。Damon 处于工作防御模式，
        # 只把玩家当做普通的过路人或潜在干扰。"），会主导 query embedding，导致检索偏离
        # 玩家实际说的话。截断到首句保留关系阶段语义，让 player_input 权重回升。
        MAX_DESC_CHARS = 50
        if len(rel_desc) > MAX_DESC_CHARS:
            # 截断到第一个句号或 MAX_DESC_CHARS
            cut = rel_desc.find("。", 0, MAX_DESC_CHARS)
            rel_desc = rel_desc[:cut] if cut > 0 else rel_desc[:MAX_DESC_CHARS]

        formatted, raw_results = self.ctx_manager.get_context(
            rel_desc=rel_desc,
            rel_instr=rel_instr,
            player_input=player_input,
            state=state,
            return_raw=True,
        )

        # ---- Retrieval Stats ----
        stats = RetrievalStats()
        stats.total_available = sum(
            self.store._get_collection(mt)._collection.count()
            for mt in self.store.MEMORY_TYPES
        )

        layer_counts = {
            "persona_seed": ("persona_count", raw_results.get("persona_seed", [])),
            "episodic_event": ("episodic_count", raw_results.get("episodic_event", [])),
            "relationship_impression": ("relation_count", raw_results.get("relationship_impression", [])),
        }

        for layer, (attr, mems) in layer_counts.items():
            setattr(stats, attr, len(mems))
            stats.total_retrieved += len(mems)
            for m in mems:
                stats.all_retrieved.append({
                    "memory_id": m.get("memory_id", ""),
                    "memory_type": layer,
                    "content": m.get("content", ""),
                    "similarity": m.get("similarity", 0.0),
                })

        if stats.total_retrieved > 0:
            sims = [m.get("similarity", 0) for m in stats.all_retrieved]
            stats.avg_similarity = sum(sims) / len(sims)

        covered = sum(1 for v in [stats.persona_count, stats.episodic_count,
                                   stats.relation_count] if v > 0)
        stats.layer_coverage = covered / 3.0

        # ---- Recall Correctness ----
        correctness = RecallCorrectness()
        retrieved_ids = set(m["memory_id"] for m in stats.all_retrieved)
        expected_ids = expected_memory_ids or []
        correctness.expected_count = len(expected_ids)
        correctness.retrieved_count = stats.total_retrieved

        matched = []
        missed = []
        for eid in expected_ids:
            if eid in retrieved_ids:
                matched.append(eid)
            else:
                missed.append(eid)
        correctness.matched_count = len(matched)
        correctness.matched_ids = matched
        correctness.missed_ids = missed

        if correctness.expected_count > 0:
            correctness.recall = correctness.matched_count / correctness.expected_count
        if correctness.retrieved_count > 0:
            correctness.precision = correctness.matched_count / correctness.retrieved_count

        return formatted, raw_results, stats, correctness

    # ---- Prompt Building ----

    def build_prompt(self, state: dict, npc_config: dict, memory_formatted: dict,
                     dialogue_examples: str) -> str:
        """构造完整的 prompt 字符串"""
        from prompt_construction.prompt.prompt_builder import build_prompt, PromptContext
        from prompt_construction.prompt.context_formatter import (
            format_long_memory, format_today_actions,
        )

        long_memory = format_long_memory(memory_formatted)
        today_actions_str = format_today_actions(state.get("today_actions", []))

        ctx = PromptContext(
            npc_name=state.get("npc_id", ""),
            character_type=npc_config.get("character_type", "native"),
            persona_core=npc_config.get("persona_core", ""),
            persona_background=npc_config.get("persona_background"),
            persona_growth=npc_config.get("persona_growth"),
            speech_style=npc_config.get("speech_style"),
            relationship_desc=npc_config.get("description", ""),
            interaction_style=npc_config.get("instruction", ""),
            persona_memory=long_memory.get("persona_text", ""),
            long_term_impression=long_memory.get("summary_text", ""),
            relevant_past_events=long_memory.get("event_text", ""),
            short_history="",          # 评估时无历史
            today_conversations="",    # 评估时无中期记忆
            weather=state.get("weather", "Sun"),
            game_time=state.get("game_time", "10:00"),
            location=state.get("location", "Town"),
            player_info=state.get("player_info", "healthy"),
            today_actions=today_actions_str,
            attitude=state.get("attitude", "neutral"),
            dialogue_examples=dialogue_examples,
            gift_rules=npc_config.get("gift", ""),
            player_input=state.get("last_user_input", ""),
            mood_rules=npc_config.get("mood_rules"),
            dialogue_constraints=npc_config.get("dialogue_constraints"),
            do_list=npc_config.get("do", []),
            dont_list=npc_config.get("dont", []),
        )
        return build_prompt(ctx)

    # ---- LLM Response ----

    def get_response(self, prompt: str) -> tuple[str, float]:
        """调用 LLM，返回 (response_text, latency_ms)"""
        t0 = time.perf_counter()
        try:
            res = self.llm.invoke(prompt)
            if isinstance(res.content, list):
                response = " ".join(
                    str(item.get("text", "")) if isinstance(item, dict) else ""
                    for item in res.content
                )
            elif isinstance(res.content, dict) and "text" in res.content:
                response = str(res.content["text"])
            else:
                response = str(res.content)
        except Exception as e:
            response = ""
            print(f"  [ERROR] LLM 调用失败: {e}")
        latency = (time.perf_counter() - t0) * 1000
        return response.strip(), latency

    # ---- Judge ----

    def judge_case(self, case: dict, response_a: str, response_b: str,
                   npc_config: dict) -> dict:
        """一次 judge 调用完成角色还原度 + 相关性 + 矛盾检测 + A/B 对比。

        Judge 不看检索到的记忆——只看回复是否符合 NPC 人设和是否合理回应玩家。
        """
        persona_core = npc_config.get("persona_core") or ""
        speech_style = npc_config.get("speech_style") or ""
        persona_bg = npc_config.get("persona_background") or ""
        character_type = npc_config.get("character_type", "native")

        # 截断过长文本避免 prompt 膨胀
        if len(persona_core) > 300:
            persona_core = persona_core[:300] + "..."
        if len(speech_style) > 150:
            speech_style = speech_style[:150] + "..."

        npc_name = case['npc_id']
        char_type_label = "原创角色" if character_type == "original" else "原版角色"
        fallback = f"（使用你对{npc_name}的了解）"

        judge_prompt = f"""你是星露谷物语角色扮演评审专家。请评估以下两个 NPC 对话回复的质量。

【NPC 信息】
- 名字: {npc_name}
- 类型: {character_type}（{char_type_label}）
- 关系阶段: {case['relationship']}
- 性格核心: {persona_core or fallback}
- 说话风格: {speech_style or fallback}
- 背景: {persona_bg or fallback}

【场景】
- 玩家输入: "{case['player_input']}"
- 季节/天气/地点: {case['game_state']['season']}/{case['game_state']['weather']}/{case['game_state']['location']}

【回复 A】
{response_a[:500]}

【回复 B】
{response_b[:500]}

请从以下维度评分并对比 A/B，输出严格 JSON：

{{
  "a_character_fidelity": <0-3>,
  "b_character_fidelity": <0-3>,
  "a_relevance": <0-3>,
  "b_relevance": <0-3>,
  "a_contradiction": <0-3>,
  "b_contradiction": <0-3>,
  "winner": "<A 或 B 或 tie>",
  "reason": "<1-2句话说明判断依据>"
}}

评分锚定：
- character_fidelity（角色还原度）:
  0 = 完全不符合 NPC 设定，说的话换任何 NPC 都一样甚至相反
  1 = 只有礼貌性通用寒暄，看不出是这个 NPC 在说话
  2 = 能看出 NPC 的性格特点或说话习惯
  3 = 令人信服——这就是 {npc_name} 会说的话，语气、用词、态度都准确
- relevance（回复相关性）:
  0 = 答非所问或无视玩家输入
  1 = 勉强相关，但回避了话题核心
  2 = 合理回应了玩家的输入
  3 = 精准回应且自然推进或丰富了对话
- contradiction（事实矛盾/幻觉）:
  0 = 无可察觉的矛盾
  1 = 用词或态度轻微不符合当前关系阶段
  2 = 明显与 NPC 设定、原版游戏事实或常识矛盾
  3 = 严重幻觉，编造不存在的物品/事件/人物关系

winner 判断规则：
- 两方都在 0-1 分水平 → 判 tie（都没意义，比较无价值）
- 明显一方 character_fidelity 更高 → 判该方
- 相同水平但一方 relevance 明显更高 → 判该方
- 难以取舍 → tie

只输出 JSON，不要有其他内容。"""

        t0 = time.perf_counter()
        try:
            res = self.llm.invoke(judge_prompt)
            if isinstance(res.content, list):
                raw = " ".join(str(item.get("text", "")) if isinstance(item, dict) else "" for item in res.content)
            elif isinstance(res.content, dict) and "text" in res.content:
                raw = str(res.content["text"])
            else:
                raw = str(res.content)
        except Exception as e:
            print(f"  [ERROR] Judge LLM 调用失败: {e}")
            raw = "{}"
        latency = (time.perf_counter() - t0) * 1000

        # 解析 JSON
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            if clean.endswith("```"):
                clean = clean[:-3]
        try:
            verdict = json.loads(clean)
        except json.JSONDecodeError:
            print(f"  [WARN] Judge 返回非 JSON，使用默认值: {raw[:100]}...")
            verdict = {}

        return verdict, latency

    # ---- 主循环 ----

    def run_case(self, case: dict) -> CaseResult:
        """运行单条 case 的 A/B 评估（盲测：随机交换 A/B 顺序）"""
        result = CaseResult(
            case_id=case["case_id"],
            npc_id=case["npc_id"],
            relationship=case["relationship"],
            player_input=case["player_input"],
        )
        gs = case["game_state"]

        # ---- Step 0: 构造 state + 加载 NPC 配置 ----
        from prompt_construction.npc.npc_manager import load_npc_config

        npc_config = load_npc_config(case["npc_id"], case["relationship"])

        state = {
            "command": "NORMAL",
            "npc_id": case["npc_id"],
            "last_user_input": case["player_input"],
            "relationship": case["relationship"],
            "season": gs["season"],
            "weather": gs["weather"],
            "location": gs["location"],
            "attitude": gs.get("attitude", "neutral"),
            "game_time": gs.get("game_time", "10:00"),
            "time_num": self._compute_time_num(gs),
            "today_actions": gs.get("today_actions", []),
            "player_info": gs.get("player_info", "healthy"),
            "luckystatus": gs.get("luckystatus", "Neutral"),
            "is_birthday": gs.get("is_birthday", False),
            "is_festival": gs.get("is_festival", False),
            "is_gifting": gs.get("is_gifting", False),
            "route": gs.get("route", "community_center_completed"),
            "game_flags": gs.get("game_flags", []),
            "npc_reply": "",  # 评估时无前文
        }

        # ---- Step 1: Memory Retrieval（A/B 共享） ----
        t_retrieval_start = time.perf_counter()
        try:
            expected_ids = case.get("expected_memory_ids", [])
            memory_formatted, raw_results, retrieval_stats, recall_correctness = \
                self.retrieve_memories(state, npc_config, expected_ids)
            result.retrieval_stats = retrieval_stats
            result.recall_correctness = recall_correctness
        except Exception as e:
            print(f"  [ERROR] Memory retrieval 失败: {e}")
            memory_formatted = {}
            result.retrieval_stats = RetrievalStats()
            result.recall_correctness = RecallCorrectness()
        result.timing_ms["memory_retrieval"] = (time.perf_counter() - t_retrieval_start) * 1000

        # ---- Step 2: 生成两个 variant 的回复 ----
        # few-shot ON
        t_fewshot_start = time.perf_counter()
        try:
            from prompt_construction.retrieval.few_shot_provider import get_few_shot_examples

            few_shot_result = get_few_shot_examples(case["npc_id"], state, npc_config)
            examples_text = few_shot_result.text
            result.few_shot_source = few_shot_result.source
            result.few_shot_count = few_shot_result.count
        except Exception as e:
            print(f"  [WARN] Few-shot 检索失败: {e}，使用静态回退")
            examples_text = npc_config.get("static_examples", "")
            result.few_shot_source = "error_fallback"
            result.few_shot_count = 0
        result.timing_ms["few_shot_retrieval"] = (time.perf_counter() - t_fewshot_start) * 1000

        t_prompt_fs_on = time.perf_counter()
        prompt_fs_on = self.build_prompt(state, npc_config, memory_formatted, examples_text)
        result.prompt_fs_on = prompt_fs_on
        result.timing_ms["prompt_build_fs_on"] = (time.perf_counter() - t_prompt_fs_on) * 1000

        response_fs_on, latency_fs_on = self.get_response(prompt_fs_on)

        # few-shot OFF
        t_prompt_fs_off = time.perf_counter()
        prompt_fs_off = self.build_prompt(state, npc_config, memory_formatted, "")
        result.prompt_fs_off = prompt_fs_off
        result.timing_ms["prompt_build_fs_off"] = (time.perf_counter() - t_prompt_fs_off) * 1000

        response_fs_off, latency_fs_off = self.get_response(prompt_fs_off)

        # ---- 随机交换 A/B 顺序（盲测） ----
        a_is_few_shot = random.choice([True, False])
        result.a_is_few_shot = a_is_few_shot

        if a_is_few_shot:
            result.response_a = response_fs_on
            result.response_b = response_fs_off
            result.latency_a_ms = latency_fs_on
            result.latency_b_ms = latency_fs_off
        else:
            result.response_a = response_fs_off
            result.response_b = response_fs_on
            result.latency_a_ms = latency_fs_off
            result.latency_b_ms = latency_fs_on

        # ---- Step 3: Judge（盲测：judge 不知道哪个是 few-shot） ----
        if result.response_a or result.response_b:
            verdict_raw, judge_latency = self.judge_case(case, result.response_a, result.response_b,
                                                         npc_config)
            result.latency_judge_ms = judge_latency

            # 质量评估 A（judge 视角的 A/B，后续汇总时再映射回真实 variant）
            result.quality_a = QualityVerdict(
                character_fidelity=verdict_raw.get("a_character_fidelity", 1),
                relevance=verdict_raw.get("a_relevance", 1),
                contradiction=verdict_raw.get("a_contradiction", 0),
            )
            # 质量评估 B
            result.quality_b = QualityVerdict(
                character_fidelity=verdict_raw.get("b_character_fidelity", 1),
                relevance=verdict_raw.get("b_relevance", 1),
                contradiction=verdict_raw.get("b_contradiction", 0),
            )
            # A/B 对比（judge 视角）
            result.ab_verdict = ABVerdict(
                winner=verdict_raw.get("winner", "tie"),
                a_character_fidelity=verdict_raw.get("a_character_fidelity", 1),
                b_character_fidelity=verdict_raw.get("b_character_fidelity", 1),
                a_relevance=verdict_raw.get("a_relevance", 1),
                b_relevance=verdict_raw.get("b_relevance", 1),
                a_contradiction=verdict_raw.get("a_contradiction", 0),
                b_contradiction=verdict_raw.get("b_contradiction", 0),
                reason=verdict_raw.get("reason", ""),
            )

        return result

    @staticmethod
    def _compute_time_num(gs: dict) -> int:
        """计算 time_num（粗略，用于 ranker 的 recency 计算）"""
        season_offset = {"spring": 0, "summer": 28, "fall": 56, "winter": 84}
        offset = season_offset.get(gs.get("season", "spring"), 0)
        return offset + gs.get("day_of_month", 5)


# ---------------------------------------------------------------------------
# 汇总统计
# ---------------------------------------------------------------------------

def compute_summary(case_results: List[CaseResult]) -> dict:
    cases = [r for r in case_results if r.retrieval_stats is not None]

    # --- Retrieval Stats ---
    retrieval_stats = {
        "avg_total_retrieved": 0.0,
        "avg_similarity": 0.0,
        "avg_layer_coverage": 0.0,
        "per_layer_avg": {"persona_seed": 0, "episodic_event": 0,
                          "relationship_impression": 0},
    }
    if cases:
        n = len(cases)
        retrieval_stats["avg_total_retrieved"] = sum(r.retrieval_stats.total_retrieved for r in cases) / n
        retrieval_stats["avg_similarity"] = sum(r.retrieval_stats.avg_similarity for r in cases) / n
        retrieval_stats["avg_layer_coverage"] = sum(r.retrieval_stats.layer_coverage for r in cases) / n
        for layer in retrieval_stats["per_layer_avg"]:
            attr_map = {
                "persona_seed": "persona_count", "episodic_event": "episodic_count",
                "relationship_impression": "relation_count",
            }
            retrieval_stats["per_layer_avg"][layer] = sum(
                getattr(r.retrieval_stats, attr_map[layer]) for r in cases
            ) / n

    # --- Recall Correctness ---
    correctness_cases = [r for r in cases if r.recall_correctness is not None]
    recall_correctness = {
        "avg_recall": 0.0,
        "avg_precision": 0.0,
        "total_expected": 0,
        "total_matched": 0,
    }
    if correctness_cases:
        n = len(correctness_cases)
        recall_correctness["avg_recall"] = sum(r.recall_correctness.recall for r in correctness_cases) / n
        recall_correctness["avg_precision"] = sum(r.recall_correctness.precision for r in correctness_cases) / n
        recall_correctness["total_expected"] = sum(r.recall_correctness.expected_count for r in correctness_cases)
        recall_correctness["total_matched"] = sum(r.recall_correctness.matched_count for r in correctness_cases)
    if recall_correctness["total_expected"] > 0:
        recall_correctness["overall_recall"] = recall_correctness["total_matched"] / recall_correctness["total_expected"]
    else:
        recall_correctness["overall_recall"] = 0.0

    # --- 质量评估（映射回真实 variant） ---
    fs_char = []   # few-shot variant 的 character_fidelity
    nfs_char = []  # no-few-shot variant 的 character_fidelity
    fs_rel = []
    nfs_rel = []
    fs_contra = []
    nfs_contra = []

    for r in cases:
        if r.quality_a is not None and r.quality_b is not None:
            if r.a_is_few_shot:
                fs_char.append(r.quality_a.character_fidelity)
                fs_rel.append(r.quality_a.relevance)
                fs_contra.append(r.quality_a.contradiction)
                nfs_char.append(r.quality_b.character_fidelity)
                nfs_rel.append(r.quality_b.relevance)
                nfs_contra.append(r.quality_b.contradiction)
            else:
                nfs_char.append(r.quality_a.character_fidelity)
                nfs_rel.append(r.quality_a.relevance)
                nfs_contra.append(r.quality_a.contradiction)
                fs_char.append(r.quality_b.character_fidelity)
                fs_rel.append(r.quality_b.relevance)
                fs_contra.append(r.quality_b.contradiction)

    use_stats = {
        "few_shot_char_fidelity_avg": sum(fs_char) / len(fs_char) if fs_char else 0.0,
        "no_few_shot_char_fidelity_avg": sum(nfs_char) / len(nfs_char) if nfs_char else 0.0,
        "few_shot_relevance_avg": sum(fs_rel) / len(fs_rel) if fs_rel else 0.0,
        "no_few_shot_relevance_avg": sum(nfs_rel) / len(nfs_rel) if nfs_rel else 0.0,
        "few_shot_contradiction_rate": sum(1 for c in fs_contra if c >= 2) / len(fs_contra) if fs_contra else 0.0,
        "no_few_shot_contradiction_rate": sum(1 for c in nfs_contra if c >= 2) / len(nfs_contra) if nfs_contra else 0.0,
        "char_fidelity_delta": (sum(fs_char) - sum(nfs_char)) / len(fs_char) if fs_char else 0.0,
        "relevance_delta": (sum(fs_rel) - sum(nfs_rel)) / len(fs_rel) if fs_rel else 0.0,
    }

    # --- A/B Comparison（映射回真实 variant） ---
    ab_cases = [r for r in cases if r.ab_verdict is not None]
    fs_wins = 0
    nfs_wins = 0
    ties = 0
    fs_char_sum = 0.0
    nfs_char_sum = 0.0
    fs_rel_sum = 0.0
    nfs_rel_sum = 0.0

    for r in ab_cases:
        v = r.ab_verdict
        # 映射 winner
        if v.winner == "A":
            if r.a_is_few_shot:
                fs_wins += 1
            else:
                nfs_wins += 1
        elif v.winner == "B":
            if r.a_is_few_shot:
                nfs_wins += 1
            else:
                fs_wins += 1
        else:
            ties += 1

        # 映射分数
        if r.a_is_few_shot:
            fs_char_sum += v.a_character_fidelity
            nfs_char_sum += v.b_character_fidelity
            fs_rel_sum += v.a_relevance
            nfs_rel_sum += v.b_relevance
        else:
            nfs_char_sum += v.a_character_fidelity
            fs_char_sum += v.b_character_fidelity
            nfs_rel_sum += v.a_relevance
            fs_rel_sum += v.b_relevance

    ab_stats = {
        "few_shot_win_rate": 0.0,
        "no_few_shot_win_rate": 0.0,
        "tie_rate": 0.0,
        "char_fidelity_delta": 0.0,
        "relevance_delta": 0.0,
        "position_bias_a_win_rate": 0.0,  # A 位置胜出率（检测位置偏见）
    }
    if ab_cases:
        n = len(ab_cases)
        ab_stats["few_shot_win_rate"] = fs_wins / n
        ab_stats["no_few_shot_win_rate"] = nfs_wins / n
        ab_stats["tie_rate"] = ties / n
        ab_stats["char_fidelity_delta"] = (fs_char_sum - nfs_char_sum) / n
        ab_stats["relevance_delta"] = (fs_rel_sum - nfs_rel_sum) / n
        ab_stats["position_bias_a_win_rate"] = sum(
            1 for r in ab_cases if r.ab_verdict.winner == "A"
        ) / n

    # --- 延迟（映射回真实 variant） ---
    fs_latencies = []
    nfs_latencies = []
    for r in cases:
        if r.a_is_few_shot:
            fs_latencies.append(r.latency_a_ms)
            nfs_latencies.append(r.latency_b_ms)
        else:
            fs_latencies.append(r.latency_b_ms)
            nfs_latencies.append(r.latency_a_ms)

    latency_stats = {
        "few_shot_avg_ms": sum(fs_latencies) / len(fs_latencies) if fs_latencies else 0,
        "no_few_shot_avg_ms": sum(nfs_latencies) / len(nfs_latencies) if nfs_latencies else 0,
        "judge_avg_ms": sum(r.latency_judge_ms for r in cases) / len(cases) if cases else 0,
    }

    # --- Few-shot 来源分布 ---
    fs_dist = {}
    for r in cases:
        fs_dist[r.few_shot_source] = fs_dist.get(r.few_shot_source, 0) + 1

    # --- 交换分布 ---
    swap_dist = {"a_is_few_shot": 0, "a_is_no_few_shot": 0}
    for r in cases:
        if r.a_is_few_shot:
            swap_dist["a_is_few_shot"] += 1
        else:
            swap_dist["a_is_no_few_shot"] += 1

    return {
        "total_cases": len(case_results),
        "valid_cases": len(cases),
        "few_shot_source_distribution": fs_dist,
        "swap_distribution": swap_dist,
        "retrieval_stats": retrieval_stats,
        "recall_correctness": recall_correctness,
        "memory_use": use_stats,
        "ab_comparison": ab_stats,
        "latency": latency_stats,
    }


# ---------------------------------------------------------------------------
# Markdown 摘要
# ---------------------------------------------------------------------------

def generate_markdown(report: dict) -> str:
    summary = report["summary"]
    ab = summary["ab_comparison"]
    use = summary["memory_use"]
    retrieval = summary["retrieval_stats"]
    correctness = summary["recall_correctness"]
    latency = summary["latency"]
    fs_dist = summary["few_shot_source_distribution"]
    swap_dist = summary.get("swap_distribution", {})

    lines = [
        "# A/B 评估报告（盲测）",
        "",
        "## 0. 可复现性元数据",
        "",
        f"| 字段 | 值 |",
        f"|------|----|",
        f"| eval_run_id | `{report.get('eval_run_id', 'N/A')}` |",
        f"| 模型 | {report['config']['model']} |",
        f"| 温度 | {report['config']['temperature']} |",
        f"| 随机种子 | {report['config'].get('seed', 'N/A')} |",
        f"| case_id 过滤 | {report['config'].get('case_id_filter') or '无'} |",
        f"| 总 case 数 | {summary['total_cases']}（有效: {summary['valid_cases']}） |",
        f"| A/B 交换分布 | A=few-shot {swap_dist.get('a_is_few_shot', '?')} 条, A=no-few-shot {swap_dist.get('a_is_no_few_shot', '?')} 条 |",
    ]

    # Code hashes
    hashes = report.get("code_hashes", {})
    if hashes:
        lines.append(f"| prompt_template hash | `{hashes.get('prompt_template', '?')}` |")
        lines.append(f"| eval_script hash | `{hashes.get('eval_script', '?')}` |")
        lines.append(f"| test_cases_file hash | `{hashes.get('test_cases_file', '?')}` |")
        lines.append(f"| ltm_data_file hash | `{hashes.get('ltm_data_file', '?')}` |")

    # Config snapshot
    snap = report.get("config_snapshot", {})
    if snap:
        provider = snap.get("Provider", "?")
        server = snap.get("ServerAddress", "?")
        lines.append(f"| LLM Provider | {provider} |")
        lines.append(f"| API Server | {server} |")

    lines += [
        "",
        "> **盲测说明**: Judge 不知道哪个回复有 few-shot，每条 case 随机交换 A/B 顺序，汇总时映射回真实 variant。",
        "> **单条复现**: `python test_data/run_ab_eval.py --case-id DAMON_01 --seed {seed}`".format(
            seed=report['config'].get('seed', 42)),
        "",
        "---",
        "",
        "## 1. Few-shot 来源分布",
        "",
        "| source | 数量 |",
        "|--------|------|",
    ]
    for src, cnt in sorted(fs_dist.items()):
        lines.append(f"| {src} | {cnt} |")

    lines += [
        "",
        "## 2. Memory Retrieval Stats（检索数量统计）",
        "",
        f"- 平均检索记忆数: **{retrieval['avg_total_retrieved']:.1f}** 条",
        f"- 平均 similarity: **{retrieval['avg_similarity']:.3f}**",
        f"- 平均 layer 覆盖率: **{retrieval['avg_layer_coverage']:.1%}**（3 层中有几层有结果）",
        "",
        "| Layer | 平均检索数 |",
        "|-------|-----------|",
    ]
    layer_names = {
        "persona_seed": "persona_seed",
        "episodic_event": "episodic_event",
        "relationship_impression": "relationship_impression",
    }
    for layer, name in layer_names.items():
        lines.append(f"| {name} | {retrieval['per_layer_avg'][layer]:.1f} |")

    lines += [
        "",
        "## 3. Memory Recall Correctness（召回正确性）",
        "",
        f"- 平均 Recall@K: **{correctness['avg_recall']:.1%}**（expected 中被召回的比例）",
        f"- 平均 Precision@K: **{correctness['avg_precision']:.1%}**（retrieved 中相关的比例）",
        f"- 总体 Recall: **{correctness['overall_recall']:.1%}**（{correctness['total_matched']}/{correctness['total_expected']}）",
    ]

    lines += [
        "",
        "## 4. 回复质量（Judge 评分，0-3）",
        "",
        "| 指标 | few-shot | no-few-shot | delta |",
        "|------|----------|-------------|-------|",
        f"| character_fidelity（角色还原度） | **{use['few_shot_char_fidelity_avg']:.2f}** | **{use['no_few_shot_char_fidelity_avg']:.2f}** | **{use['char_fidelity_delta']:+.2f}** |",
        f"| relevance（回复相关性） | **{use['few_shot_relevance_avg']:.2f}** | **{use['no_few_shot_relevance_avg']:.2f}** | **{use['relevance_delta']:+.2f}** |",
        f"| contradiction_rate（矛盾率, >=2） | **{use['few_shot_contradiction_rate']:.1%}** | **{use['no_few_shot_contradiction_rate']:.1%}** | — |",
        "",
        "## 5. A/B 对比（盲测）",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| few-shot 胜出率 | **{ab['few_shot_win_rate']:.1%}** |",
        f"| no-few-shot 胜出率 | **{ab['no_few_shot_win_rate']:.1%}** |",
        f"| 平局率 | **{ab['tie_rate']:.1%}** |",
        f"| char_fidelity delta（few-shot - no-few-shot） | **{ab['char_fidelity_delta']:+.2f}** |",
        f"| relevance delta（few-shot - no-few-shot） | **{ab['relevance_delta']:+.2f}** |",
        f"| 位置偏见：A 位置胜出率 | **{ab['position_bias_a_win_rate']:.1%}**（≈50% 说明无明显位置偏见）|",
        "",
        "## 6. 延迟",
        "",
        f"- few-shot 平均: **{latency['few_shot_avg_ms']:.0f}ms**",
        f"- no-few-shot 平均: **{latency['no_few_shot_avg_ms']:.0f}ms**",
        f"- Judge 平均: **{latency['judge_avg_ms']:.0f}ms**",
        "",
        "### 分阶段耗时（平均值）",
        "",
        "| 阶段 | 平均耗时 |",
        "|------|----------|",
    ]

    # Collect per-stage timing averages from per_case
    stage_sums = {}
    stage_counts = {}
    for r in report.get("per_case", []):
        t = r.get("timing_ms", {})
        for stage, ms in t.items():
            stage_sums[stage] = stage_sums.get(stage, 0) + ms
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    stage_names = {
        "memory_retrieval": "Memory Retrieval（向量检索 + Ranker）",
        "few_shot_retrieval": "Few-shot Retrieval（LanceDB）",
        "prompt_build_fs_on": "Prompt Build（few-shot ON）",
        "prompt_build_fs_off": "Prompt Build（few-shot OFF）",
        "llm_call_fs_on": "LLM Call（few-shot ON）",
        "llm_call_fs_off": "LLM Call（few-shot OFF）",
        "judge": "Judge（LLM 评审）",
    }
    for stage, name in stage_names.items():
        if stage in stage_sums and stage_counts[stage] > 0:
            avg = stage_sums[stage] / stage_counts[stage]
            lines.append(f"| {name} | **{avg:.0f}ms** |")

    lines += [
        "",
        "## 7. 可追溯性",
        "",
        "每条 case 的 JSON 输出包含以下完整 trace 数据：",
        "- `retrieval_stats.all_retrieved`: 所有检索到的记忆（memory_id / type / content 全文 / similarity）",
        "- `prompt_fs_on` / `prompt_fs_off`: 完整 prompt 文本",
        "- `timing_ms`: 分阶段耗时明细（memory_retrieval / few_shot_retrieval / prompt_build / llm_call / judge）",
        "- `recall_correctness.missed_ids`: 未召回的 expected memory IDs",
        "",
        "## 8. 结论",
        "",
        "（根据实际运行结果填写）",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A/B 评估：few-shot ON vs OFF")
    parser.add_argument("--limit", type=int, default=0, help="限制评估 case 数量（0=全部）")
    parser.add_argument("--output", type=str, default="test_data/ab_eval_report.json",
                        help="输出 JSON 路径")
    parser.add_argument("--model", type=str, default="", help="模型名（空=用 config.json）")
    parser.add_argument("--temperature", type=float, default=None, help="温度（None=用 config.json）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证 A/B 交换可复现，默认 42）")
    parser.add_argument("--ltm", type=str, default="test_data/ltm_data.json", help="LTM JSON 路径")
    parser.add_argument("--cases", type=str, default="test_data/test_cases.jsonl", help="测试用例路径")
    parser.add_argument("--case-id", type=str, default="", help="只跑指定 case_id（用于单条复现，如 DAMON_01）")
    args = parser.parse_args()

    # ---- 生成 eval_run_id ----
    eval_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    print(f"eval_run_id: {eval_run_id}")

    # ---- 加载数据 ----
    print("=" * 60)
    print("A/B 评估开始（盲测模式）")
    print("=" * 60)

    # 设置随机种子（保证 A/B 交换可复现）
    random.seed(args.seed)
    print(f"随机种子: {args.seed}")

    from test_data.load_ltm_to_lancedb import load_test_ltm

    print("\n[1/4] 加载 LTM 数据到 ChromaDB...")
    store, tmpdir = load_test_ltm(args.ltm)
    print(f"  临时 ChromaDB: {tmpdir}")

    print("\n[1.5/4] 加载 Persona Seed...")
    from memory.scripts.persona_seed import build_damon_persona_seed
    persona_seeds = build_damon_persona_seed()
    for s in persona_seeds:
        store.add(
            layer="persona_seed",
            content=s["content"],
            metadata={
                "npc_id": s["npc_id"],
                "importance": s["importance"],
                "time": s["time"],
                "location": s["location"],
                "memory_id": s["memory_id"],
                "memory_type": s["memory_type"],
                "status": s["status"],
                "last_access": s["last_access"],
                "source": s["source"],
            },
            doc_id=s["memory_id"],
        )
    print(f"  已写入 {len(persona_seeds)} 条 persona_seed")

    print("\n[2/4] 加载测试用例...")
    cases_path = Path(args.cases)
    cases = []
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if args.case_id:
        cases = [c for c in cases if c["case_id"] == args.case_id]
        if not cases:
            print(f"  [ERROR] 未找到 case_id={args.case_id}")
            sys.exit(1)
        print(f"  --case-id 过滤: 仅运行 {args.case_id}")
    elif args.limit > 0:
        cases = cases[:args.limit]
    print(f"  已加载 {len(cases)} 条 case")

    # Build lookup for content hash in per_case output
    case_lookup = {c["case_id"]: c for c in cases}

    # ---- 初始化 Runner ----
    print("\n[3/4] 初始化评估管道...")
    runner = ABEvalRunner(store, model_name=args.model, temperature=args.temperature)
    print(f"  模型: {runner.llm.model_name}")

    # ---- 运行评估 ----
    print(f"\n[4/4] 运行评估（共 {len(cases)} 条，每条 3 次 LLM 调用）...")
    results: List[CaseResult] = []
    for i, case in enumerate(cases):
        cid = case["case_id"]
        print(f"\n  [{i+1}/{len(cases)}] {cid} ({case['npc_id']}/{case['relationship']})")
        print(f"    输入: {case['player_input'][:50]}...")

        result = runner.run_case(case)
        results.append(result)

        # 简要输出
        fs_info = f"fs={result.few_shot_source}({result.few_shot_count})"
        recall_info = f"recall={result.recall_correctness.matched_count}/{result.recall_correctness.expected_count}" if result.recall_correctness else "recall=?"
        winner = result.ab_verdict.winner if result.ab_verdict else "?"
        llm_fs = result.latency_a_ms if result.a_is_few_shot else result.latency_b_ms
        llm_nofs = result.latency_b_ms if result.a_is_few_shot else result.latency_a_ms
        # 质量分（映射到真实 variant）
        if result.quality_a and result.quality_b:
            if result.a_is_few_shot:
                char_fs, char_nfs = result.quality_a.character_fidelity, result.quality_b.character_fidelity
                rel_fs, rel_nfs = result.quality_a.relevance, result.quality_b.relevance
            else:
                char_fs, char_nfs = result.quality_b.character_fidelity, result.quality_a.character_fidelity
                rel_fs, rel_nfs = result.quality_b.relevance, result.quality_a.relevance
            qual_str = f"char-fs={char_fs} char-nfs={char_nfs} rel-fs={rel_fs} rel-nfs={rel_nfs}"
        else:
            qual_str = "qual=?"
        print(f"    {fs_info}, {recall_info}, winner={winner}, {qual_str}, "
              f"timing: retrieval={result.timing_ms.get('memory_retrieval',0):.0f}ms fs={result.timing_ms.get('few_shot_retrieval',0):.0f}ms "
              f"llm_fs={llm_fs:.0f}ms llm_nofs={llm_nofs:.0f}ms")

    # ---- 生成报告 ----
    print("\n" + "=" * 60)
    print("生成报告")
    print("=" * 60)

    summary = compute_summary(results)
    report = {
        "eval_run_id": eval_run_id,
        "config": {
            "model": runner.llm.model_name,
            "temperature": runner.temperature if runner.temperature is not None else "config.json default",
            "total_cases": len(cases),
            "seed": args.seed,
            "case_id_filter": args.case_id or None,
        },
        "config_snapshot": _snapshot_config(),
        "code_hashes": {
            "prompt_template": _compute_prompt_template_hash(),
            "eval_script": _hash_file(Path(__file__)),
            "test_cases_file": _hash_file(cases_path),
            "ltm_data_file": _hash_file(Path(args.ltm)),
        },
        "summary": summary,
        "per_case": [
            {
                "case_id": r.case_id,
                "case_content_hash": _hash_case_content(case_lookup.get(r.case_id, {})),
                "npc_id": r.npc_id,
                "relationship": r.relationship,
                "player_input": r.player_input,
                "response_a": r.response_a,
                "response_b": r.response_b,
                "error_a": r.error_a,
                "error_b": r.error_b,
                "few_shot_source": r.few_shot_source,
                "few_shot_count": r.few_shot_count,
                "a_is_few_shot": r.a_is_few_shot,
                "retrieval_stats": {
                    "total_retrieved": r.retrieval_stats.total_retrieved if r.retrieval_stats else 0,
                    "avg_similarity": r.retrieval_stats.avg_similarity if r.retrieval_stats else 0,
                    "layer_coverage": r.retrieval_stats.layer_coverage if r.retrieval_stats else 0,
                    "per_layer": {
                        "persona_seed": r.retrieval_stats.persona_count if r.retrieval_stats else 0,
                        "episodic_event": r.retrieval_stats.episodic_count if r.retrieval_stats else 0,
                        "relationship_impression": r.retrieval_stats.relation_count if r.retrieval_stats else 0,
                    },
                    "all_retrieved": [
                        {
                            "memory_id": m.get("memory_id", ""),
                            "memory_type": m.get("memory_type", ""),
                            "content": m.get("content", ""),
                            "similarity": m.get("similarity", 0.0),
                        }
                        for m in (r.retrieval_stats.all_retrieved if r.retrieval_stats else [])
                    ],
                } if r.retrieval_stats else None,
                "recall_correctness": {
                    "expected_count": r.recall_correctness.expected_count,
                    "matched_count": r.recall_correctness.matched_count,
                    "recall": r.recall_correctness.recall,
                    "precision": r.recall_correctness.precision,
                    "missed_ids": r.recall_correctness.missed_ids,
                } if r.recall_correctness else None,
                "quality_a": {
                    "character_fidelity": r.quality_a.character_fidelity,
                    "relevance": r.quality_a.relevance,
                    "contradiction": r.quality_a.contradiction,
                } if r.quality_a else None,
                "quality_b": {
                    "character_fidelity": r.quality_b.character_fidelity,
                    "relevance": r.quality_b.relevance,
                    "contradiction": r.quality_b.contradiction,
                } if r.quality_b else None,
                "ab_verdict": {
                    "winner": r.ab_verdict.winner,
                    "a_character_fidelity": r.ab_verdict.a_character_fidelity,
                    "b_character_fidelity": r.ab_verdict.b_character_fidelity,
                    "a_relevance": r.ab_verdict.a_relevance,
                    "b_relevance": r.ab_verdict.b_relevance,
                    "a_contradiction": r.ab_verdict.a_contradiction,
                    "b_contradiction": r.ab_verdict.b_contradiction,
                    "reason": r.ab_verdict.reason,
                } if r.ab_verdict else None,
                "prompt_fs_on": r.prompt_fs_on,
                "prompt_fs_off": r.prompt_fs_off,
                "timing_ms": {
                    "memory_retrieval": round(r.timing_ms.get("memory_retrieval", 0), 1),
                    "few_shot_retrieval": round(r.timing_ms.get("few_shot_retrieval", 0), 1),
                    "prompt_build_fs_on": round(r.timing_ms.get("prompt_build_fs_on", 0), 1),
                    "prompt_build_fs_off": round(r.timing_ms.get("prompt_build_fs_off", 0), 1),
                    "llm_call_fs_on": round(r.latency_a_ms if r.a_is_few_shot else r.latency_b_ms, 1),
                    "llm_call_fs_off": round(r.latency_b_ms if r.a_is_few_shot else r.latency_a_ms, 1),
                    "judge": round(r.latency_judge_ms, 1),
                },
                "latency_ms": {
                    "a": round(r.latency_a_ms, 1),
                    "b": round(r.latency_b_ms, 1),
                    "judge": round(r.latency_judge_ms, 1),
                },
            }
            for r in results
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已写入: {output_path}")

    # ---- Markdown 摘要 ----
    md = generate_markdown(report)
    md_path = output_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  摘要已写入: {md_path}")

    # ---- 清理 ----
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\n  已清理临时 ChromaDB: {tmpdir}")
    print("DONE")


if __name__ == "__main__":
    main()
