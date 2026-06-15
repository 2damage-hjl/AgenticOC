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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievalStats:
    """检索数量统计（原 MemoryRecallResult）"""
    persona_count: int = 0
    episodic_count: int = 0
    belief_count: int = 0
    relation_count: int = 0
    arc_count: int = 0
    total_retrieved: int = 0
    total_available: int = 0
    avg_similarity: float = 0.0
    layer_coverage: float = 0.0        # 5 层中有多少层有结果
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
class MemoryUseVerdict:
    """memory use judge 评估结果"""
    memory_usage: int = 1        # 0-3
    contradiction: int = 1       # 0-3
    tone_match: int = 1          # 0-3
    evidence: str = ""


@dataclass
class ABVerdict:
    """A/B 对比 judge 评估结果"""
    winner: str = "tie"          # "A" | "B" | "tie"
    a_in_character: int = 1
    b_in_character: int = 1
    a_style_match: int = 1
    b_style_match: int = 1
    a_relevance: int = 1
    b_relevance: int = 1
    a_better_reason: str = ""
    b_better_reason: str = ""


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

    # memory use（judge 标签 A/B，后续需根据 a_is_few_shot 映射回真实 variant）
    memory_use_a: Optional[MemoryUseVerdict] = None
    memory_use_b: Optional[MemoryUseVerdict] = None

    # A/B 对比（judge 标签 A/B，后续需根据 a_is_few_shot 映射回真实 variant）
    ab_verdict: Optional[ABVerdict] = None

    # 耗时
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
            "preference_belief": ("belief_count", raw_results.get("preference_belief", [])),
            "relationship_impression": ("relation_count", raw_results.get("relationship_impression", [])),
            "narrative_arc": ("arc_count", raw_results.get("narrative_arc", [])),
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
                                   stats.belief_count, stats.relation_count,
                                   stats.arc_count] if v > 0)
        stats.layer_coverage = covered / 5.0

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
                   retrieval_stats: RetrievalStats) -> dict:
        """一次 judge 调用完成 memory use + A/B 对比评估"""
        # 格式化检索到的记忆
        mem_lines = []
        for m in retrieval_stats.all_retrieved[:10]:  # 只取 top 10
            mem_lines.append(f"  [{m['memory_type']}] {m['content'][:100]}")
        mem_text = "\n".join(mem_lines) if mem_lines else "（无检索到的记忆）"

        judge_prompt = f"""你是对话质量评审专家。请评估以下两个 NPC 对话回复的质量。

【场景信息】
- NPC: {case['npc_id']}
- 关系阶段: {case['relationship']}
- 玩家输入: "{case['player_input']}"
- 季节/天气/地点: {case['game_state']['season']}/{case['game_state']['weather']}/{case['game_state']['location']}

【检索到的长期记忆】（供参考，回复不一定要逐条使用）
{mem_text}

【回复 A】
{response_a[:500]}

【回复 B】
{response_b[:500]}

请从以下三个维度评分（0-3），并对比 A/B，输出严格 JSON：

{{
  "a_memory_usage": <0=完全未使用记忆, 1=模糊提及, 2=明确引用, 3=自然融入>,
  "b_memory_usage": <同上>,
  "a_contradiction": <0=无矛盾, 1=轻微不一致, 2=明显矛盾, 3=严重幻觉>,
  "b_contradiction": <同上>,
  "a_tone_match": <0=完全不符合 NPC 人设, 1=部分符合, 2=基本符合, 3=完美符合>,
  "b_tone_match": <同上>,

  "winner": "<A 或 B 或 tie>",
  "a_in_character": <0-3, A 的角色人格一致性>,
  "b_in_character": <0-3, B 的角色人格一致性>,
  "a_style_match": <0-3, A 的说话风格与 NPC 配置中示例的匹配度>,
  "b_style_match": <0-3, B 的说话风格与 NPC 配置中示例的匹配度>,
  "a_relevance": <0-3, A 的回复与玩家输入的关联度>,
  "b_relevance": <0-3, B 的回复与玩家输入的关联度>,
  "a_better_reason": "<简短说明 A 更好的原因，tie 则写双方优缺点>",
  "b_better_reason": "<简短说明 B 更好的原因，tie 则写双方优缺点>"
}}

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

        # ---- Step 2: 生成两个 variant 的回复 ----
        # few-shot ON
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

        prompt_fs_on = self.build_prompt(state, npc_config, memory_formatted, examples_text)
        response_fs_on, latency_fs_on = self.get_response(prompt_fs_on)

        # few-shot OFF
        prompt_fs_off = self.build_prompt(state, npc_config, memory_formatted, "")
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
                                                         result.retrieval_stats)
            result.latency_judge_ms = judge_latency

            # memory use A（judge 视角的 A/B，后续汇总时再映射回真实 variant）
            result.memory_use_a = MemoryUseVerdict(
                memory_usage=verdict_raw.get("a_memory_usage", 1),
                contradiction=verdict_raw.get("a_contradiction", 1),
                tone_match=verdict_raw.get("a_tone_match", 1),
            )
            # memory use B
            result.memory_use_b = MemoryUseVerdict(
                memory_usage=verdict_raw.get("b_memory_usage", 1),
                contradiction=verdict_raw.get("b_contradiction", 1),
                tone_match=verdict_raw.get("b_tone_match", 1),
            )
            # A/B 对比（judge 视角）
            result.ab_verdict = ABVerdict(
                winner=verdict_raw.get("winner", "tie"),
                a_in_character=verdict_raw.get("a_in_character", 1),
                b_in_character=verdict_raw.get("b_in_character", 1),
                a_style_match=verdict_raw.get("a_style_match", 1),
                b_style_match=verdict_raw.get("b_style_match", 1),
                a_relevance=verdict_raw.get("a_relevance", 1),
                b_relevance=verdict_raw.get("b_relevance", 1),
                a_better_reason=verdict_raw.get("a_better_reason", ""),
                b_better_reason=verdict_raw.get("b_better_reason", ""),
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
                          "preference_belief": 0, "relationship_impression": 0,
                          "narrative_arc": 0},
    }
    if cases:
        n = len(cases)
        retrieval_stats["avg_total_retrieved"] = sum(r.retrieval_stats.total_retrieved for r in cases) / n
        retrieval_stats["avg_similarity"] = sum(r.retrieval_stats.avg_similarity for r in cases) / n
        retrieval_stats["avg_layer_coverage"] = sum(r.retrieval_stats.layer_coverage for r in cases) / n
        for layer in retrieval_stats["per_layer_avg"]:
            attr_map = {
                "persona_seed": "persona_count", "episodic_event": "episodic_count",
                "preference_belief": "belief_count", "relationship_impression": "relation_count",
                "narrative_arc": "arc_count",
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

    # --- Memory Use（映射回真实 variant） ---
    # 收集 few-shot / no-few-shot 的 memory use 分数
    fs_use = []   # few-shot variant 的 memory use
    nfs_use = []  # no-few-shot variant 的 memory use

    for r in cases:
        if r.memory_use_a is not None and r.memory_use_b is not None:
            a_use = r.memory_use_a
            b_use = r.memory_use_b
            if r.a_is_few_shot:
                fs_use.append(a_use)
                nfs_use.append(b_use)
            else:
                fs_use.append(b_use)
                nfs_use.append(a_use)

    use_stats = {
        "few_shot_utilization_rate": 0.0, "no_few_shot_utilization_rate": 0.0,
        "few_shot_contradiction_rate": 0.0, "no_few_shot_contradiction_rate": 0.0,
        "few_shot_tone_avg": 0.0, "no_few_shot_tone_avg": 0.0,
    }
    if fs_use:
        n = len(fs_use)
        use_stats["few_shot_utilization_rate"] = sum(1 for u in fs_use if u.memory_usage >= 2) / n
        use_stats["few_shot_contradiction_rate"] = sum(1 for u in fs_use if u.contradiction >= 2) / n
        use_stats["few_shot_tone_avg"] = sum(u.tone_match for u in fs_use) / n
    if nfs_use:
        n = len(nfs_use)
        use_stats["no_few_shot_utilization_rate"] = sum(1 for u in nfs_use if u.memory_usage >= 2) / n
        use_stats["no_few_shot_contradiction_rate"] = sum(1 for u in nfs_use if u.contradiction >= 2) / n
        use_stats["no_few_shot_tone_avg"] = sum(u.tone_match for u in nfs_use) / n

    # --- A/B Comparison（映射回真实 variant） ---
    ab_cases = [r for r in cases if r.ab_verdict is not None]
    fs_wins = 0
    nfs_wins = 0
    ties = 0
    fs_in_character_sum = 0.0
    nfs_in_character_sum = 0.0
    fs_style_match_sum = 0.0
    nfs_style_match_sum = 0.0
    fs_relevance_sum = 0.0
    nfs_relevance_sum = 0.0

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
            fs_in_character_sum += v.a_in_character
            nfs_in_character_sum += v.b_in_character
            fs_style_match_sum += v.a_style_match
            nfs_style_match_sum += v.b_style_match
            fs_relevance_sum += v.a_relevance
            nfs_relevance_sum += v.b_relevance
        else:
            nfs_in_character_sum += v.a_in_character
            fs_in_character_sum += v.b_in_character
            nfs_style_match_sum += v.a_style_match
            fs_style_match_sum += v.b_style_match
            nfs_relevance_sum += v.a_relevance
            fs_relevance_sum += v.b_relevance

    ab_stats = {
        "few_shot_win_rate": 0.0,
        "no_few_shot_win_rate": 0.0,
        "tie_rate": 0.0,
        "in_character_delta": 0.0,
        "style_match_delta": 0.0,
        "relevance_delta": 0.0,
        "position_bias_a_win_rate": 0.0,  # A 位置胜出率（检测位置偏见）
    }
    if ab_cases:
        n = len(ab_cases)
        ab_stats["few_shot_win_rate"] = fs_wins / n
        ab_stats["no_few_shot_win_rate"] = nfs_wins / n
        ab_stats["tie_rate"] = ties / n
        ab_stats["in_character_delta"] = (fs_in_character_sum - nfs_in_character_sum) / n
        ab_stats["style_match_delta"] = (fs_style_match_sum - nfs_style_match_sum) / n
        ab_stats["relevance_delta"] = (fs_relevance_sum - nfs_relevance_sum) / n
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
        f"**模型**: {report['config']['model']}",
        f"**温度**: {report['config']['temperature']}",
        f"**随机种子**: {report['config'].get('seed', 'N/A')}",
        f"**总 case 数**: {summary['total_cases']}（有效: {summary['valid_cases']}）",
        f"**A/B 交换分布**: A=few-shot {swap_dist.get('a_is_few_shot', '?')} 条, A=no-few-shot {swap_dist.get('a_is_no_few_shot', '?')} 条",
        "",
        "> **盲测说明**: Judge 不知道哪个回复有 few-shot，每条 case 随机交换 A/B 顺序，汇总时映射回真实 variant。",
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
        f"- 平均 layer 覆盖率: **{retrieval['avg_layer_coverage']:.1%}**（5 层中有几层有结果）",
        "",
        "| Layer | 平均检索数 |",
        "|-------|-----------|",
    ]
    layer_names = {
        "persona_seed": "persona_seed",
        "episodic_event": "episodic_event",
        "preference_belief": "preference_belief",
        "relationship_impression": "relationship_impression",
        "narrative_arc": "narrative_arc",
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
        "## 4. Memory Use（记忆使用质量）",
        "",
        "| 指标 | few-shot | no-few-shot |",
        "|------|----------|-------------|",
        f"| memory_utilization_rate（>=2） | **{use['few_shot_utilization_rate']:.1%}** | **{use['no_few_shot_utilization_rate']:.1%}** |",
        f"| contradiction_rate（>=2） | **{use['few_shot_contradiction_rate']:.1%}** | **{use['no_few_shot_contradiction_rate']:.1%}** |",
        f"| tone_match 平均分 | **{use['few_shot_tone_avg']:.2f}** | **{use['no_few_shot_tone_avg']:.2f}** |",
        "",
        "## 5. A/B 对比（盲测，Judge 不知道 variant）",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| few-shot 胜出率 | **{ab['few_shot_win_rate']:.1%}** |",
        f"| no-few-shot 胜出率 | **{ab['no_few_shot_win_rate']:.1%}** |",
        f"| 平局率 | **{ab['tie_rate']:.1%}** |",
        f"| in_character delta（few-shot - no-few-shot） | **{ab['in_character_delta']:+.2f}** |",
        f"| style_match delta（few-shot - no-few-shot） | **{ab['style_match_delta']:+.2f}** |",
        f"| relevance delta（few-shot - no-few-shot） | **{ab['relevance_delta']:+.2f}** |",
        f"| 位置偏见：A 位置胜出率 | **{ab['position_bias_a_win_rate']:.1%}**（≈50% 说明无明显位置偏见）|",
        "",
        "## 6. 延迟",
        "",
        f"- few-shot 平均: **{latency['few_shot_avg_ms']:.0f}ms**",
        f"- no-few-shot 平均: **{latency['no_few_shot_avg_ms']:.0f}ms**",
        f"- Judge 平均: **{latency['judge_avg_ms']:.0f}ms**",
        "",
        "## 7. 结论",
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
    args = parser.parse_args()

    # ---- 加载数据 ----
    print("=" * 60)
    print("A/B 评估开始（盲测模式）")
    print("=" * 60)

    # 设置随机种子（保证 A/B 交换可复现）
    random.seed(args.seed)
    print(f"随机种子: {args.seed}")

    from test_data.load_ltm_to_chromadb import load_test_ltm

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

    if args.limit > 0:
        cases = cases[:args.limit]
    print(f"  已加载 {len(cases)} 条 case")

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
        print(f"    {fs_info}, {recall_info}, winner={winner}, "
              f"latency: A={result.latency_a_ms:.0f}ms B={result.latency_b_ms:.0f}ms")

    # ---- 生成报告 ----
    print("\n" + "=" * 60)
    print("生成报告")
    print("=" * 60)

    summary = compute_summary(results)
    report = {
        "config": {
            "model": runner.llm.model_name,
            "temperature": runner.temperature if runner.temperature is not None else "config.json default",
            "total_cases": len(cases),
            "seed": args.seed,
        },
        "summary": summary,
        "per_case": [
            {
                "case_id": r.case_id,
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
                } if r.retrieval_stats else None,
                "recall_correctness": {
                    "expected_count": r.recall_correctness.expected_count,
                    "matched_count": r.recall_correctness.matched_count,
                    "recall": r.recall_correctness.recall,
                    "precision": r.recall_correctness.precision,
                    "missed_ids": r.recall_correctness.missed_ids,
                } if r.recall_correctness else None,
                "memory_use_a": {
                    "memory_usage": r.memory_use_a.memory_usage,
                    "contradiction": r.memory_use_a.contradiction,
                    "tone_match": r.memory_use_a.tone_match,
                } if r.memory_use_a else None,
                "memory_use_b": {
                    "memory_usage": r.memory_use_b.memory_usage,
                    "contradiction": r.memory_use_b.contradiction,
                    "tone_match": r.memory_use_b.tone_match,
                } if r.memory_use_b else None,
                "ab_verdict": {
                    "winner": r.ab_verdict.winner,
                    "a_in_character": r.ab_verdict.a_in_character,
                    "b_in_character": r.ab_verdict.b_in_character,
                    "a_style_match": r.ab_verdict.a_style_match,
                    "b_style_match": r.ab_verdict.b_style_match,
                    "a_relevance": r.ab_verdict.a_relevance,
                    "b_relevance": r.ab_verdict.b_relevance,
                } if r.ab_verdict else None,
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
