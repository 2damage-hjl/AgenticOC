"""
生成用于 few-shot 质量评估的测试数据。

根据 plan.md 要求：
- 5 个重点 NPC: Damon, Abigail, Shane, Sebastian, Penny
- 每个 NPC 10 条测试 case，总计 50 条
- LTM 数据覆盖全部 5 个 collection

人称约定（对齐实际代码）：
- persona_seed：第三人称客观描述（代码里是静态设定，非 NPC 自述）
- episodic_event：NPC 第一人称（MTMemory.py prompt: "以 NPC 第一人称描述"）
- preference_belief：NPC 第一人称（MTMemory.py prompt: "以 NPC 第一人称描述，如'玩家似乎不喜欢紫水晶'"）
- relationship_impression：第三人称（程序计算值 + 格式化字符串）
- narrative_arc：NPC 第一人称内心独白（consolidation.py prompt: "用第一人称，写内心独白式周记"）

输出：
- test_data/ltm_data.json       — 长期记忆数据（按 NPC + collection 组织）
- test_data/test_cases.jsonl    — 50 条对话测试用例（JSONL 格式）
"""

import hashlib
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


def _content_hash(content: str, length: int = 8) -> str:
    """确定性哈希，保证 memory_id 在内容不变时稳定。"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:length]


def make_memory(memory_type, npc_id, content, time, location, importance,
                status="active", source="test_data", **extra):
    prefix_map = {
        "persona_seed": "persona",
        "episodic_event": "evt",
        "preference_belief": "belief",
        "relationship_impression": "rel",
        "narrative_arc": "arc",
    }
    prefix = prefix_map.get(memory_type, "mem")
    mem = {
        "memory_id": f"{prefix}_{npc_id}_{_content_hash(content)}",
        "memory_type": memory_type,
        "npc_id": npc_id,
        "content": content,
        "time": time,
        "location": location,
        "importance": importance,
        "status": status,
        "last_access": 0 if time == "static" else time,
        "source": source,
    }
    mem.update(extra)
    return mem


def make_test_case(case_id, npc_id, relationship, player_input, season, weather,
                   location, attitude="neutral", day_of_month=5, year=1,
                   game_time="10:00", today_actions=None, is_birthday=False,
                   is_festival=False, is_gifting=False, route="community_center_completed",
                   game_flags=None, player_info="healthy", luckystatus="Neutral",
                   expected_ltm_context=None):
    return {
        "case_id": case_id,
        "npc_id": npc_id,
        "relationship": relationship,
        "player_input": player_input,
        "game_state": {
            "season": season, "weather": weather, "location": location,
            "attitude": attitude, "day_of_month": day_of_month, "year": year,
            "game_time": game_time,
            "today_actions": today_actions or [],
            "is_birthday": is_birthday, "is_festival": is_festival,
            "is_gifting": is_gifting, "route": route,
            "game_flags": game_flags or [],
            "player_info": player_info, "luckystatus": luckystatus,
        },
        "expected_ltm_context": expected_ltm_context or [],
    }


# ============================================================
# Damon LTM
# ============================================================

def generate_damon_ltm():
    npc = "Damon"
    memories = []

    # --- persona_seed（第三人称）---
    memories.append(make_memory("persona_seed", npc,
        "性格核心：慢热、外冷内热。不熟悉时礼貌疏离，熟悉后会展露脆弱和冷幽默。",
        "static", "persona_seed", 0.9, source="init"))
    memories.append(make_memory("persona_seed", npc,
        "背景：来自祖祖城，受刘易斯镇长委托负责海上宾馆建设项目。暂时在鹈鹕镇附近租房，白天考察小镇，常去图书馆和镇长家。",
        "static", "persona_seed", 0.85, source="init"))
    memories.append(make_memory("persona_seed", npc,
        "说话风格：不熟悉时句子简短、用敬语、保持距离。熟悉后语气放松、夹杂冷幽默。傲娇时会用$4表情。",
        "static", "persona_seed", 0.8, source="init"))
    memories.append(make_memory("persona_seed", npc,
        "成长弧线：前期勤勉但麻木的中产阶级，因工作居无定所，怀疑人生。与玩家成为挚友后，被鹈鹕镇治愈，找到平静，视这里为归宿。",
        "static", "persona_seed", 0.85, source="init"))
    memories.append(make_memory("persona_seed", npc,
        "工作态度：极度看重专业素养，对项目进度有强迫症般的执着。年底赶项目的焦虑已经成为身体记忆。",
        "static", "persona_seed", 0.7, source="init"))
    memories.append(make_memory("persona_seed", npc,
        "对鹈鹕镇的态度：从最初的'只是工作地点'逐渐转变为'也许可以留下来的地方'。镇上的人情味让他感到陌生但温暖。",
        "static", "persona_seed", 0.75, source="init"))

    # --- episodic_event（NPC 第一人称）---
    memories.append(make_memory("episodic_event", npc,
        "有个自称农场主的人来海滩找我搭话。我正在看地形图，只是礼貌性地点了点头——工作期间不想被打扰。",
        1, "Beach", 0.3, topic_tags=["first_meeting", "work"]))

    memories.append(make_memory("episodic_event", npc,
        "雨下得很大，玩家居然给我带了一杯热咖啡。我有点意外，说了句'谢谢，这比图书馆的速溶好多了'。这个人挺细心的。",
        3, "Town", 0.5, topic_tags=["gift", "coffee", "rainy"],
        emotional_valence=0.3))

    memories.append(make_memory("episodic_event", npc,
        "我在图书馆查海滨建筑的案例，玩家过来问需不需要帮忙。我婉拒了——这是我的工作，不想欠人情。但说实话，有人主动帮忙的感觉不坏。",
        5, "Library", 0.35, topic_tags=["work", "offer_help"]))

    memories.append(make_memory("episodic_event", npc,
        "我在酒馆角落喝啤酒看文件，玩家坐过来了。不知道为什么，我居然跟他吐槽了审批流程的繁琐——平时我不会跟人说这些的。",
        8, "Saloon", 0.55, topic_tags=["work", "complaint", "saloon"],
        emotional_valence=-0.2))

    memories.append(make_memory("episodic_event", npc,
        "在镇广场碰到玩家，我主动问了农场经营的事。说实话，如果以后项目需要从农场采购原材料，我会优先考虑他——他的农场看起来打理得不错。",
        12, "Town", 0.6, topic_tags=["work", "farm", "business"],
        emotional_valence=0.4))

    memories.append(make_memory("episodic_event", npc,
        "在海边测绘的时候玩家过来了。我指着远处跟他说海上宾馆大概在那个位置，然后让他保密。我居然会跟一个本地农场主分享项目细节……大概是开始信任他了。",
        15, "Beach", 0.65, topic_tags=["work", "secret", "beach"],
        emotional_valence=0.5))

    memories.append(make_memory("episodic_event", npc,
        "我居然主动问玩家有没有什么怪癖，然后跟他说我焦虑的时候会不自觉抠手指。在别人面前承认这种事真是丢脸。$4",
        20, "Saloon", 0.7, topic_tags=["vulnerability", "anxiety", "personal"],
        emotional_valence=0.6))

    memories.append(make_memory("episodic_event", npc,
        "冬夜在山间散步时碰到了玩家。我问他'你有没有觉得，夜晚的鹈鹕镇比白天更真实？'我们沉默地走了一段路。这种不说话也不尴尬的感觉，很久没有过了。",
        25, "Mountain", 0.75, topic_tags=["night", "deep_talk", "winter"],
        emotional_valence=0.5))

    memories.append(make_memory("episodic_event", npc,
        "我在亚历克斯那买了两个冰淇淋，给了玩家一个。我嘴硬说是留一个明天吃——其实我就是想给他买。$4",
        30, "Saloon", 0.8, topic_tags=["gift", "tsundere", "food"],
        emotional_valence=0.8))

    memories.append(make_memory("episodic_event", npc,
        "我对玩家说了一句很丢脸的话：'我以前觉得安定下来是一种放弃。现在觉得，也许这才是我一直在找的东西。'说出口以后反而轻松了。",
        35, "Beach", 0.85, topic_tags=["growth", "reflection", "belonging"],
        emotional_valence=0.9))

    memories.append(make_memory("episodic_event", npc,
        "秋天傍晚在山间对玩家说'秋天的山谷……我开始觉得这里比城市好了'。说完我自己都愣了一下——以前我绝对不会说这种话。",
        28, "Mountain", 0.75, topic_tags=["season", "fall", "growth"],
        emotional_valence=0.7))

    memories.append(make_memory("episodic_event", npc,
        "在图书馆发呆的时候想到了祖祖城。我跟玩家说'如果当初没接这个项目，我大概还在写字楼里加班到凌晨'。说出来以后发现，好像没那么怀念了。",
        22, "Library", 0.7, topic_tags=["past", "reflection", "city_life"],
        emotional_valence=0.3))

    # --- preference_belief（NPC 第一人称）---
    memories.append(make_memory("preference_belief", npc,
        "玩家好像挺喜欢喝咖啡的，至少那次接过咖啡的时候表情是高兴的。",
        3, "Town", 0.6, target="player", belief_key=f"{npc}.player.preference.coffee",
        belief_type="player_preference", topic="coffee", polarity="like",
        confidence=0.75))

    memories.append(make_memory("preference_belief", npc,
        "玩家对农场经营很上心，聊到农场的时候明显很投入。应该是真心喜欢种地这件事。",
        12, "Town", 0.65, target="player", belief_key=f"{npc}.player.preference.farm_work",
        belief_type="player_preference", topic="farm_work", polarity="like",
        confidence=0.80))

    memories.append(make_memory("preference_belief", npc,
        "玩家似乎对城市生活没什么好感。我聊起祖祖城的时候，他的反应让我觉得他也不喜欢那种日子。",
        22, "Library", 0.55, target="player", belief_key=f"{npc}.player.preference.city_life",
        belief_type="player_preference", topic="city_life", polarity="dislike",
        confidence=0.65))

    memories.append(make_memory("preference_belief", npc,
        "不太确定玩家喜不喜欢海鲜。在海边聊到海上宾馆的时候他没什么特别的反应。",
        15, "Beach", 0.4, target="player", belief_key=f"{npc}.player.preference.seafood",
        belief_type="player_preference", topic="seafood", polarity="uncertain_neutral",
        confidence=0.45))

    # --- relationship_impression（第三人称结构化数据）---
    memories.append(make_memory("relationship_impression", npc,
        f"Damon 对 player 的长期印象：信任 0.60，亲近 0.55，熟悉 0.50，困惑 0.15，怨气 0.00。",
        35, "Beach", 0.85, target="player", impression_key=f"{npc}.player.relationship",
        trust=0.60, warmth=0.55, familiarity=0.50, confusion=0.15, resentment=0.00))

    # --- narrative_arc（NPC 第一人称内心独白）---
    memories.append(make_memory("narrative_arc", npc,
        "这两周我基本把玩家当成一个普通的农场主，对话也仅限于工作上的寒暄。有一天他给我带了杯热咖啡，比图书馆的速溶好喝——这是我第一次觉得这个镇上的人不一样。",
        14, "narrative", 0.6, week_range="第1-2周"))

    memories.append(make_memory("narrative_arc", npc,
        "我开始主动跟玩家聊农场的事，甚至在海滩上跟他说了宾馆选址的秘密。我为什么会跟他说这些？大概是觉得他不会到处乱讲吧。我还开始跟他开一些冷玩笑——以前在祖祖城我从来不开玩笑的。",
        28, "narrative", 0.7, week_range="第3-4周"))

    memories.append(make_memory("narrative_arc", npc,
        "我在玩家面前几乎没什么保留了。我跟他说了焦虑、说了抠手指的坏习惯、说了对城市生活的厌倦。他甚至看穿了我用'买两个冰淇淋留一个明天吃'当借口。我以前觉得安定下来是放弃，现在觉得……也许这才是我一直在找的东西。这个小镇、这些人、还有他。",
        42, "narrative", 0.8, week_range="第5-6周"))

    return memories


# ============================================================
# Abigail LTM
# ============================================================

def generate_abigail_ltm():
    npc = "Abigail"
    memories = []

    # 注意：Abigail 是原版 NPC，系统中无 persona_seed，角色信息来自 NPC JSON + LanceDB 对话数据

    # --- episodic_event（NPC 第一人称）---
    memories.append(make_memory("episodic_event", npc,
        "那个新来的农场主跟我搭话的时候，我正因为周末没写作业而烦躁，语气不太友善。希望他没往心里去。",
        1, "Town", 0.3, topic_tags=["first_meeting", "homework"]))

    memories.append(make_memory("episodic_event", npc,
        "玩家路过山上的时候停下来听我吹长笛。我吹完有点不好意思，说了句'我只是随便吹吹'——其实我很高兴有人愿意听。",
        4, "Mountain", 0.5, topic_tags=["flute", "music", "shy"],
        emotional_valence=0.3))

    memories.append(make_memory("episodic_event", npc,
        "下雨天站在湖边看雨的时候玩家过来了。我跟他说'看着雨帘在寂静的湖面上飘荡……这种感觉难以形容'。他好像能理解这种感觉。",
        7, "Mountain", 0.55, topic_tags=["rainy", "lake", "contemplation"],
        emotional_valence=0.2))

    memories.append(make_memory("episodic_event", npc,
        "我问玩家有没有去过矿洞探险，然后拿出一块紫水晶给他看。这是我亲自在矿洞里找到的，我觉得很漂亮——不知道他懂不懂。",
        10, "Town", 0.6, topic_tags=["mines", "amethyst", "adventure"],
        emotional_valence=0.5))

    memories.append(make_memory("episodic_event", npc,
        "我跟玩家吐槽了和爸妈吵架的事。我知道他们是好意，但他们就是不能理解我。玩家好像听得很认真，没有打断我。",
        13, "Town", 0.65, topic_tags=["family", "parents", "conflict"],
        emotional_valence=-0.3))

    memories.append(make_memory("episodic_event", npc,
        "玩家送了我一个南瓜！我问他今年种南瓜了没有，让他帮我留一个——我最喜欢南瓜了。这个农场主真有意思。",
        16, "Town", 0.6, topic_tags=["gift", "pumpkin", "fall"],
        emotional_valence=0.7))

    memories.append(make_memory("episodic_event", npc,
        "我跟玩家说昨晚去海边看到奇怪的光在晃动。可能只是梦吧，但感觉不太真实。他听完没有笑我——大多数人会笑我的。",
        19, "Beach", 0.65, topic_tags=["supernatural", "beach", "dream"],
        emotional_valence=0.3))

    memories.append(make_memory("episodic_event", npc,
        "在墓地碰到玩家的时候我问他'要是在墓地里待一整夜会发生什么事呢？'我当然是带着神秘笑容说的。他看起来居然有点感兴趣。",
        22, "Town", 0.6, topic_tags=["graveyard", "mystery", "adventure"],
        emotional_valence=0.4))

    memories.append(make_memory("episodic_event", npc,
        "我对玩家说'你这人真有趣，你能搬来这里真是太好了'。这是我第一次直接夸他——说出来以后心跳得很快。",
        25, "Town", 0.7, topic_tags=["friendship", "compliment"],
        emotional_valence=0.8))

    memories.append(make_memory("episodic_event", npc,
        "我最近一直在做白日梦……玩家问我梦到什么了。我说那是个秘密。其实梦到的是他，但我不好意思说。",
        28, "Mountain", 0.7, topic_tags=["secret", "daydream", "romance"],
        emotional_valence=0.6))

    memories.append(make_memory("episodic_event", npc,
        "我鼓起勇气想夸玩家，结果说出来的却是'你的靴子真挺干净的'。我在心里把自己骂了一百遍。$4",
        32, "Town", 0.8, topic_tags=["romance", "awkward", "compliment"],
        emotional_valence=0.9))

    memories.append(make_memory("episodic_event", npc,
        "我昨晚做了个美梦，梦到了玩家。我告诉他了——然后假装去看远处的风景。",
        35, "Farm", 0.8, topic_tags=["dream", "romance", "intimate"],
        emotional_valence=0.9))

    memories.append(make_memory("episodic_event", npc,
        "秋天在农场和玩家一起看落叶。蘑菇、腐烂的树叶、南瓜的香味混在一起，我对他说'很美好不是吗？'他点头的时候，我觉得这一刻会记很久。",
        38, "Farm", 0.75, topic_tags=["fall", "farm", "peaceful"],
        emotional_valence=0.8))

    # --- preference_belief（NPC 第一人称）---
    memories.append(make_memory("preference_belief", npc,
        "玩家应该挺喜欢挖矿的。每次聊到矿洞探险他就很有兴致，跟我一样。",
        10, "Town", 0.6, target="player", belief_key=f"{npc}.player.preference.mining",
        belief_type="player_preference", topic="mining", polarity="like",
        confidence=0.80))

    memories.append(make_memory("preference_belief", npc,
        "玩家好像很喜欢紫水晶。我给他看的时候他眼睛都亮了——这让我特别高兴。",
        10, "Town", 0.65, target="player", belief_key=f"{npc}.player.preference.amethyst",
        belief_type="player_preference", topic="amethyst", polarity="like",
        confidence=0.85))

    memories.append(make_memory("preference_belief", npc,
        "我送玩家南瓜的时候他很开心，他应该也喜欢南瓜。秋天是南瓜的季节嘛。",
        16, "Town", 0.55, target="player", belief_key=f"{npc}.player.preference.pumpkin",
        belief_type="player_preference", topic="pumpkin", polarity="like",
        confidence=0.70))

    memories.append(make_memory("preference_belief", npc,
        "不太确定玩家对夏天是什么感觉。有一次我抱怨夏天太闷热的时候他没什么特别反应。",
        4, "Mountain", 0.35, target="player", belief_key=f"{npc}.player.preference.summer",
        belief_type="player_preference", topic="summer", polarity="uncertain_neutral",
        confidence=0.40))

    # --- relationship_impression（第三人称）---
    memories.append(make_memory("relationship_impression", npc,
        f"Abigail 对 player 的长期印象：信任 0.70，亲近 0.75，熟悉 0.65，困惑 0.20，怨气 0.00。",
        38, "Farm", 0.85, target="player", impression_key=f"{npc}.player.relationship",
        trust=0.70, warmth=0.75, familiarity=0.65, confusion=0.20, resentment=0.00))

    # --- narrative_arc（NPC 第一人称）---
    memories.append(make_memory("narrative_arc", npc,
        "这两周我对那个新来的农场主没什么特别印象。有次在山上吹长笛的时候他停下来听了——我有点不好意思，但心里其实挺高兴的。他好像不是那种无聊的大人。",
        14, "narrative", 0.55, week_range="第1-2周"))

    memories.append(make_memory("narrative_arc", npc,
        "我开始跟玩家聊一些真正在意的事——矿洞探险、紫水晶、和爸妈吵架的烦恼。他还送了我一个南瓜！我不知道为什么愿意跟他说这么多。也许是因为他不打断我，也不对我说教。",
        28, "narrative", 0.7, week_range="第3-4周"))

    memories.append(make_memory("narrative_arc", npc,
        "我觉得我在意玩家，超出了普通朋友的范畴。我做了梦，梦里有他。我试着夸他漂亮，结果说出口的却是'你的靴子真干净'——我到底在干什么啊！但他说不定觉得这样的我很可爱呢？",
        42, "narrative", 0.8, week_range="第5-6周"))

    return memories


# ============================================================
# Shane LTM
# ============================================================

def generate_shane_ltm():
    npc = "Shane"
    memories = []

    # 注意：Shane 是原版 NPC，系统中无 persona_seed，角色信息来自 NPC JSON + LanceDB 对话数据

    # --- episodic_event（NPC 第一人称）---
    memories.append(make_memory("episodic_event", npc,
        "有个新来的农场主跟我搭话。我不认识他，为什么要跟我说话？我跟他说了句'我不认识你'就走开了。",
        2, "Town", 0.3, topic_tags=["first_meeting", "rude", "rejection"]))

    memories.append(make_memory("episodic_event", npc,
        "玩家在JojaMart碰到我在上货。我头都没抬——你要买东西就去柜台，别跟我说话。我上班已经够烦了。",
        5, "JojaMart", 0.35, topic_tags=["work", "cold", "joja"]))

    memories.append(make_memory("episodic_event", npc,
        "傍晚在牧场外面淋雨的时候被玩家看到了。我让他走开，但说实话……那一刻我其实有点不想一个人待着。",
        8, "Forest", 0.5, topic_tags=["rainy", "vulnerable", "depression"],
        emotional_valence=-0.5))

    memories.append(make_memory("episodic_event", npc,
        "这个农场主每天都来找我说话。真是够锲而不舍的。我今天终于跟他说了句'真没想到还有人愿意和我说话'——可能是真话。",
        14, "Town", 0.6, topic_tags=["persistence", "breakthrough", "surprise"],
        emotional_valence=0.2))

    memories.append(make_memory("episodic_event", npc,
        "玩家在酒馆请我喝了杯啤酒。我说'谢了……这比一个人喝强一点'。真的，强一点。",
        17, "Saloon", 0.6, topic_tags=["gift", "beer", "saloon"],
        emotional_valence=0.4))

    memories.append(make_memory("episodic_event", npc,
        "我带玩家去看我的鸡了。给他介绍了查理——它是我最喜欢的一只。我跟他说'它们不会嫌我烦'。说这话的时候我居然笑了一下。",
        21, "Forest", 0.7, topic_tags=["chickens", "charlie", "passion"],
        emotional_valence=0.6))

    memories.append(make_memory("episodic_event", npc,
        "在酒馆又喝多了。我居然对着玩家说了'有时候我觉得活着真累……你为什么要一直来找我？难道你不觉得我很烦吗？'第二天醒来后悔死了。",
        24, "Saloon", 0.75, topic_tags=["drunk", "vulnerable", "depression"],
        emotional_valence=-0.7))

    memories.append(make_memory("episodic_event", npc,
        "昨晚又喝醉了。但玩家没有走，陪我坐到了酒馆打烊。第二天我跟他说'昨晚的事别跟别人说'——其实我想说谢谢。",
        25, "Saloon", 0.8, topic_tags=["support", "trust", "vulnerable"],
        emotional_valence=0.3))

    memories.append(make_memory("episodic_event", npc,
        "我对玩家说'我戒酒了——大概吧。至少今天没喝'。说出口的那一刻，我自己都信了。也许我真的可以做到。",
        30, "Forest", 0.8, topic_tags=["sobriety", "growth", "effort"],
        emotional_valence=0.5))

    memories.append(make_memory("episodic_event", npc,
        "我跟玩家说了'如果你是认真的，那我就是全世界最幸运的人'。然后立刻转移话题说鸡舍要打扫了——我怕他看出我说的每个字都是认真的。",
        35, "Farm", 0.85, topic_tags=["romance", "awkward", "gratitude"],
        emotional_valence=0.9))

    memories.append(make_memory("episodic_event", npc,
        "下雨天和玩家待在家里。我跟他说'雨天在家……有你在就好。我以前从来不敢想这种生活'。说完觉得鼻子有点酸。",
        40, "Farm", 0.8, topic_tags=["rainy", "domestic", "gratitude"],
        emotional_valence=0.8))

    memories.append(make_memory("episodic_event", npc,
        "早上给玩家泡了咖啡，什么都没说就放在他旁边。'……早。'——这是我表达关心的方式。他应该懂吧。",
        42, "Farm", 0.7, topic_tags=["morning", "coffee", "care"],
        emotional_valence=0.7))

    # --- preference_belief（NPC 第一人称）---
    memories.append(make_memory("preference_belief", npc,
        "玩家在酒馆喝过我请的啤酒，看起来不排斥。也许他也喜欢偶尔来一杯。",
        17, "Saloon", 0.5, target="player", belief_key=f"{npc}.player.preference.beer",
        belief_type="player_preference", topic="beer", polarity="like",
        confidence=0.65))

    memories.append(make_memory("preference_belief", npc,
        "玩家来看我的鸡的时候很感兴趣，不是那种假装的。他真的觉得查理很可爱——这点我很确定。",
        21, "Forest", 0.6, target="player", belief_key=f"{npc}.player.preference.chickens",
        belief_type="player_preference", topic="chickens", polarity="like",
        confidence=0.75))

    memories.append(make_memory("preference_belief", npc,
        "我做过一次辣椒爆炒——玩家吃了以后表情很满足。他应该是喜欢这个的。",
        30, "Forest", 0.6, target="player", belief_key=f"{npc}.player.preference.pepper_poppers",
        belief_type="player_preference", topic="pepper_poppers", polarity="like",
        confidence=0.80))

    # --- relationship_impression（第三人称）---
    memories.append(make_memory("relationship_impression", npc,
        f"Shane 对 player 的长期印象：信任 0.65，亲近 0.60，熟悉 0.70，困惑 0.25，怨气 0.00。",
        42, "Farm", 0.85, target="player", impression_key=f"{npc}.player.relationship",
        trust=0.65, warmth=0.60, familiarity=0.70, confusion=0.25, resentment=0.00))

    # --- narrative_arc（NPC 第一人称）---
    memories.append(make_memory("narrative_arc", npc,
        "这两周我一直在赶走那个新来的农场主。'走开'、'别烦我'——我每次都这么说。但他第二天又来了。为什么？一般人早就放弃了。这个人要么是特别固执，要么是脑子有问题。",
        14, "narrative", 0.5, week_range="第1-2周"))

    memories.append(make_memory("narrative_arc", npc,
        "我不再赶玩家走了。带他看了我的鸡——这连我自己都没想到。但在酒馆喝醉后，我把最难听的话都说给了他听。醒来后我以为他会消失，但他没有。他只是说'昨晚的事别跟别人说'。这个人到底图什么？",
        28, "narrative", 0.65, week_range="第3-4周"))

    memories.append(make_memory("narrative_arc", npc,
        "我开始戒酒了——不是完全成功，但至少今天没喝。玩家说他为我感到骄傲，我差点哭出来。我以前总觉得像我这样的人不配拥有好日子，但现在我觉得……也许我可以试试。为了他，也为了我自己。",
        42, "narrative", 0.75, week_range="第5-6周"))

    return memories


# ============================================================
# Sebastian LTM
# ============================================================

def generate_sebastian_ltm():
    npc = "Sebastian"
    memories = []

    # 注意：Sebastian 是原版 NPC，系统中无 persona_seed，角色信息来自 NPC JSON + LanceDB 对话数据

    # --- episodic_event（NPC 第一人称）---
    memories.append(make_memory("episodic_event", npc,
        "新来的农场主闯进地下室找我说话。我头都没抬——那么多地方你不选，偏偏选中鹈鹕镇？行吧。",
        1, "Mountain", 0.3, topic_tags=["first_meeting", "skeptical"]))

    memories.append(make_memory("episodic_event", npc,
        "雨天的山上碰到了玩家。我说了句'运气好的话，这种天气可能会看见青蛙'。比平时说的话多了一点——大概是下雨让人放松。",
        5, "Mountain", 0.45, topic_tags=["rainy", "frogs", "nature"]))

    memories.append(make_memory("episodic_event", npc,
        "晚上在湖边抽烟的时候玩家过来了。我跟他说'我通常只有天黑后才出去'——然后问他觉不觉得奇怪。他说不觉得。",
        9, "Mountain", 0.5, topic_tags=["night", "smoking", "loner"],
        emotional_valence=0.1))

    memories.append(make_memory("episodic_event", npc,
        "我不小心问出口了——'要是我突然消失，会有人挂念我吗？'说出口就后悔了。这种问题不应该问别人。",
        14, "Mountain", 0.65, topic_tags=["existential", "vulnerable", "loneliness"],
        emotional_valence=-0.3))

    memories.append(make_memory("episodic_event", npc,
        "和山姆在酒馆打台球的时候玩家也在。打台球的时候我比平时放松，可能还笑了一下——山姆有这种效果。",
        17, "Saloon", 0.55, topic_tags=["friends", "sam", "pool"],
        emotional_valence=0.4))

    memories.append(make_memory("episodic_event", npc,
        "我跟玩家吐槽了玛鲁。我说大家都喜欢她不过是因为她会吸引注意力——说完立刻道歉了。其实我知道这不公平，玛鲁没那么坏。但有时候就是忍不住。",
        20, "Mountain", 0.65, topic_tags=["family", "maru", "jealousy", "honest"],
        emotional_valence=-0.2))

    memories.append(make_memory("episodic_event", npc,
        "雨天的海滩只有我和玩家。我对着远处的海平线说了一堆平时不会说的话——'遥望天水一线的苍茫让我有动力继续向前走'。他听完什么都没说，但我知道他懂了。",
        24, "Beach", 0.75, topic_tags=["rainy", "beach", "deep_talk", "motivation"],
        emotional_valence=0.5))

    memories.append(make_memory("episodic_event", npc,
        "我对玩家说了实话：'你跟山姆是我在这镇上唯一的朋友'。说完我立刻低头看手机——说这种话太不像我了。",
        28, "Mountain", 0.75, topic_tags=["friendship", "gratitude", "rare_admission"],
        emotional_valence=0.7))

    memories.append(make_memory("episodic_event", npc,
        "夜里看着满身是泥的玩家，我居然说了一句'你一直看起来很棒，即使在院子里弄得满身都是泥之后'。好吧，这就是我的情话水平。",
        33, "Farm", 0.8, topic_tags=["romance", "compliment", "night"],
        emotional_valence=0.8))

    memories.append(make_memory("episodic_event", npc,
        "我认真地跟玩家说'我在尽最大努力戒烟……我不想比你先去世。这是个坏习惯。我想和你共创未来'。我以前从来不会想'未来'这种事的。",
        38, "Farm", 0.85, topic_tags=["growth", "smoking", "future", "love"],
        emotional_valence=0.9))

    memories.append(make_memory("episodic_event", npc,
        "被噩梦惊醒后睡不着，给玩家煮了咖啡。他问我还好吗，我说了实话——'从噩梦中醒来，再也睡不着了'。在他面前说这些好像没那么难。",
        41, "Farm", 0.7, topic_tags=["morning", "coffee", "nightmare", "care"],
        emotional_valence=0.6))

    memories.append(make_memory("episodic_event", npc,
        "冬夜跟玩家说'和你住在这里让我学会了走出自己的堡垒。我觉得这样对我很好'。说这种话还是有点不好意思，但他在笑，所以应该值得。",
        44, "Farm", 0.8, topic_tags=["winter", "growth", "gratitude"],
        emotional_valence=0.8))

    # --- preference_belief（NPC 第一人称）---
    memories.append(make_memory("preference_belief", npc,
        "玩家好像也挺喜欢雨天的。在雨天的海滩上他看起来比平时放松——跟我一样。",
        24, "Beach", 0.55, target="player", belief_key=f"{npc}.player.preference.rain",
        belief_type="player_preference", topic="rain", polarity="like",
        confidence=0.70))

    memories.append(make_memory("preference_belief", npc,
        "玩家跟我和山姆一起玩过游戏。他应该是喜欢这类的，至少不排斥。",
        17, "Saloon", 0.55, target="player", belief_key=f"{npc}.player.preference.video_games",
        belief_type="player_preference", topic="video_games", polarity="like",
        confidence=0.75))

    memories.append(make_memory("preference_belief", npc,
        "我提过 Solarian Chronicles 的时候玩家表现出兴趣，不是敷衍的那种。也许他也会喜欢跑团？",
        28, "Mountain", 0.5, target="player", belief_key=f"{npc}.player.preference.solarian_chronicles",
        belief_type="player_preference", topic="solarian_chronicles", polarity="like",
        confidence=0.60))

    memories.append(make_memory("preference_belief", npc,
        "不太确定玩家对摩托车的态度。我说过想骑车带他但他说了句'看看吧'——可能是有点紧张。",
        33, "Farm", 0.45, target="player", belief_key=f"{npc}.player.preference.motorcycle",
        belief_type="player_preference", topic="motorcycle", polarity="uncertain_neutral",
        confidence=0.50))

    # --- relationship_impression（第三人称）---
    memories.append(make_memory("relationship_impression", npc,
        f"Sebastian 对 player 的长期印象：信任 0.70，亲近 0.65，熟悉 0.70，困惑 0.10，怨气 0.00。",
        44, "Farm", 0.85, target="player", impression_key=f"{npc}.player.relationship",
        trust=0.70, warmth=0.65, familiarity=0.70, confusion=0.10, resentment=0.00))

    # --- narrative_arc（NPC 第一人称）---
    memories.append(make_memory("narrative_arc", npc,
        "这两周我对那个新来的农场主基本无感。他搬来鹈鹕镇的时候我就在想——那么多地方不选，选这里？不过有次雨天在山上碰到，我说了句关于青蛙的话，他好像真的在听。",
        14, "narrative", 0.5, week_range="第1-2周"))

    memories.append(make_memory("narrative_arc", npc,
        "玩家越来越频繁地出现在我的生活里。我不小心跟他说了'如果我消失了会有人挂念吗'这种话——太丢人了。但后来我居然还跟他吐槽了玛鲁的事。他听完没教训我，只是安静地听。这让我觉得……也许我可以跟这个人说更多。",
        28, "narrative", 0.65, week_range="第3-4周"))

    memories.append(make_memory("narrative_arc", npc,
        "我人生中有两件重要的事发生了：第一，玩家成了我生命中最重要的人。第二，我开始戒烟、开始想'未来'、开始走出那间地下室。跟他说'和你住在这里让我学会了走出自己的堡垒'的时候，我是认真的。虽然我还是需要很多独处的时间，但他好像懂了。",
        44, "narrative", 0.8, week_range="第5-6周"))

    return memories


# ============================================================
# Penny LTM
# ============================================================

def generate_penny_ltm():
    npc = "Penny"
    memories = []

    # 注意：Penny 是原版 NPC，系统中无 persona_seed，角色信息来自 NPC JSON + LanceDB 对话数据

    # --- episodic_event（NPC 第一人称）---
    memories.append(make_memory("episodic_event", npc,
        "新来的农场主跟我打招呼了。我说'哦……你好！我是潘妮……'然后就不知道说什么了。我应该多说一点的。",
        1, "Town", 0.3, topic_tags=["first_meeting", "shy", "introduction"]))

    memories.append(make_memory("episodic_event", npc,
        "玩家路过博物馆的时候我在给贾斯和文森特上课。我跟他讲了我们在学什么——教孩子们读书是我最开心的事。他好像也在认真地听。",
        4, "Town", 0.5, topic_tags=["teaching", "children", "passion"],
        emotional_valence=0.5))

    memories.append(make_memory("episodic_event", npc,
        "下雨天我在图书馆的角落里看书，玩家坐到了我旁边。我跟他说'图书馆比家里安静得多'——说完有点后悔，不该跟别人抱怨家里的。但他没有追问。",
        8, "Library", 0.55, topic_tags=["rainy", "library", "escape"],
        emotional_valence=0.1))

    memories.append(make_memory("episodic_event", npc,
        "我居然跟玩家说了妈妈酗酒的事。我说'有时候我会担心妈妈……虽然她不会承认自己需要人担心'。说完我立刻道歉了——这种事不应该跟别人说的。但他说没关系，他理解。",
        13, "Town", 0.7, topic_tags=["pam", "alcohol", "family", "worry"],
        emotional_valence=-0.4))

    memories.append(make_memory("episodic_event", npc,
        "我跟玩家说了想带孩子们去森林郊游的事，说要找一个熟悉自然的人当特邀讲师。其实我就是在想他——他经营农场，对自然最了解了。但我不好意思直接说。",
        17, "Town", 0.6, topic_tags=["teaching", "field_trip", "enthusiasm"],
        emotional_valence=0.6))

    memories.append(make_memory("episodic_event", npc,
        "我忍不住跟玩家分享了我的白日梦：'如果能开个小花园就好了……也许有一天我也能住在有花园的地方。也许我会住在农场呢！'然后我咯咯笑了——其实是认真的。",
        21, "Town", 0.65, topic_tags=["dream", "garden", "farm", "hope"],
        emotional_valence=0.5))

    memories.append(make_memory("episodic_event", npc,
        "万灵节的时候我和孩子们一起做了南瓜灯。我跟玩家说'秋天的时候孩子们特别期待万灵节——我也是'。他也在笑，那个画面让我觉得好温暖。",
        25, "Town", 0.55, topic_tags=["fall", "festival", "children", "joy"],
        emotional_valence=0.7))

    memories.append(make_memory("episodic_event", npc,
        "我对玩家说了'谢谢你愿意听我说这些'。是真的——我从来没有跟别人说过这么多心里话。他看着我的眼神让我觉得，跟他说这些是对的。",
        28, "Library", 0.7, topic_tags=["gratitude", "trust", "vulnerable"],
        emotional_valence=0.7))

    memories.append(make_memory("episodic_event", npc,
        "我对玩家说'我一直觉得人生不可能像小说那样精彩，可是现在真的有那种感觉了'。说出这句话的时候我脸红了，但我没有后悔。",
        32, "Town", 0.75, topic_tags=["romance", "happiness", "novel"],
        emotional_valence=0.9))

    memories.append(make_memory("episodic_event", npc,
        "婚前几天我紧张得不行。我跟玩家说'我有些害怕。这是天翻地覆的变化！别担心，我真的很高兴'。高兴是真的，害怕也是真的——但害怕的那种感觉，和以前做噩梦时的害怕不一样。",
        36, "Town", 0.75, topic_tags=["wedding", "nervous", "excited"],
        emotional_valence=0.7))

    memories.append(make_memory("episodic_event", npc,
        "在新家的第一个月，我对玩家说'这里真宁静。我过去经常做最恐怖的噩梦，但是现在我睡得很香'。这是真的。噩梦再也没有来过。",
        39, "Farm", 0.8, topic_tags=["home", "peace", "nightmares", "healing"],
        emotional_valence=0.9))

    memories.append(make_memory("episodic_event", npc,
        "早上给玩家做了早餐。我说'不是很好看，但是很有营养的'——虽然还在学习中，但能为他做饭这件事本身就让我很幸福。",
        42, "Farm", 0.7, topic_tags=["morning", "breakfast", "care"],
        emotional_valence=0.8))

    # --- preference_belief（NPC 第一人称）---
    memories.append(make_memory("preference_belief", npc,
        "玩家在图书馆的时候很安静地读书。我觉得他是真的享受阅读，不是装出来的。",
        8, "Library", 0.55, target="player", belief_key=f"{npc}.player.preference.reading",
        belief_type="player_preference", topic="reading", polarity="like",
        confidence=0.75))

    memories.append(make_memory("preference_belief", npc,
        "玩家跟孩子们在一起的时候特别温和，贾斯和文森特都喜欢他。他应该是真心喜欢小孩的。",
        17, "Town", 0.6, target="player", belief_key=f"{npc}.player.preference.children",
        belief_type="player_preference", topic="children", polarity="like",
        confidence=0.80))

    memories.append(make_memory("preference_belief", npc,
        "我做的早餐虽然不太好看，但玩家每次都吃得很干净。他应该是喜欢我做的饭的——或者至少不讨厌吧。",
        42, "Farm", 0.5, target="player", belief_key=f"{npc}.player.preference.cooking",
        belief_type="player_preference", topic="cooking", polarity="like",
        confidence=0.65))

    # --- relationship_impression（第三人称）---
    memories.append(make_memory("relationship_impression", npc,
        f"Penny 对 player 的长期印象：信任 0.80，亲近 0.75，熟悉 0.70，困惑 0.05，怨气 0.00。",
        42, "Farm", 0.85, target="player", impression_key=f"{npc}.player.relationship",
        trust=0.80, warmth=0.75, familiarity=0.70, confusion=0.05, resentment=0.00))

    # --- narrative_arc（NPC 第一人称）---
    memories.append(make_memory("narrative_arc", npc,
        "这两周我认识了新来的农场主。他很礼貌，我也尽量礼貌地回应了——虽然每次说完'你好'我就不知道该说什么了。有次他在博物馆看到我上课，我觉得他好像对教育也挺感兴趣的。",
        14, "narrative", 0.5, week_range="第1-2周"))

    memories.append(make_memory("narrative_arc", npc,
        "我跟玩家说了很多——比跟任何人说的都多。关于妈妈酗酒的事、关于想在农场有个花园的白日梦、关于孩子们。他每次都很认真地听，从不打断我。我开始觉得，也许我不用总是把所有事情都憋在心里。",
        28, "narrative", 0.65, week_range="第3-4周"))

    memories.append(make_memory("narrative_arc", npc,
        "我的人生好像翻开了一本新书。我跟玩家在一起了，搬进了农场，拥有了自己的家——一个真的有花园的家。我不再做噩梦了。以前我觉得人生不可能像小说那样精彩，但我错了。每天早上醒来给他做早餐的时候，我觉得这就是我一直梦想的生活。",
        42, "narrative", 0.8, week_range="第5-6周"))

    return memories


# ============================================================
# 测试用例（不变，仅保留 expected_ltm_context 供人工参考）
# ============================================================

def generate_test_cases():
    cases = []

    # ========== Damon（18 条）==========
    cases.append(make_test_case("DAMON_01", "Damon", "stranger",
        "你好，你就是新来的那个建筑师吗？", "spring", "Sun", "Town",
        game_time="10:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 工作态度、对鹈鹕镇的态度",
                              "episodic_event: first_meeting/海滩"]))

    cases.append(make_test_case("DAMON_02", "Damon", "stranger",
        "在忙什么呢？需要帮忙吗？", "winter", "Rain", "Mountain",
        game_time="14:00", today_actions=["mining", "mining"],
        expected_ltm_context=["persona_seed: 工作态度、说话风格",
                              "episodic_event: 无显著匹配（stranger 阶段事件少）"]))

    cases.append(make_test_case("DAMON_03", "Damon", "acquaintance",
        "最近项目进展怎么样？", "summer", "Sun", "Beach",
        game_time="11:00", today_actions=["fishing"],
        expected_ltm_context=["persona_seed: 工作态度",
                              "episodic_event: 海边测绘/保密分享",
                              "preference_belief: seafood（不确定）"]))

    cases.append(make_test_case("DAMON_04", "Damon", "acquaintance",
        "我给你带了杯咖啡。", "fall", "Rain", "Town",
        game_time="09:00", today_actions=["farming", "foraging"], is_gifting=True,
        expected_ltm_context=["persona_seed: 说话风格",
                              "episodic_event: 送咖啡/雨天",
                              "preference_belief: coffee（like）"]))

    cases.append(make_test_case("DAMON_05", "Damon", "friend",
        "你在图书馆看什么书呢？", "spring", "Sun", "Library",
        game_time="15:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 背景",
                              "episodic_event: 图书馆/工作/婉拒帮助",
                              "preference_belief: city_life（dislike）"]))

    cases.append(make_test_case("DAMON_06", "Damon", "friend",
        "今天辛苦了，我请你喝一杯吧。", "summer", "Rain", "Saloon",
        game_time="19:00", today_actions=["farming", "fishing", "mining"], is_gifting=True,
        expected_ltm_context=["persona_seed: 说话风格/冷幽默",
                              "episodic_event: 酒馆/吐槽审批/放松",
                              "preference_belief: coffee（like）"]))

    cases.append(make_test_case("DAMON_07", "Damon", "close friend",
        "你还好吗？看起来有点累。", "fall", "Sun", "Beach",
        game_time="16:00", today_actions=["fishing", "foraging"],
        expected_ltm_context=["persona_seed: 成长弧线",
                              "episodic_event: 焦虑/抠手指/脆弱",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("DAMON_08", "Damon", "close friend",
        "夜晚的山里真安静啊。", "winter", "Sun", "Mountain",
        game_time="22:00", today_actions=["mining"],
        expected_ltm_context=["persona_seed: 成长弧线",
                              "episodic_event: 冬夜/深谈/'夜晚更真实'",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("DAMON_09", "Damon", "best friend",
        "你有后悔过来鹈鹕镇吗？", "spring", "Sun", "Town",
        game_time="12:00", today_actions=["farming", "farming"],
        expected_ltm_context=["persona_seed: 成长弧线、对鹈鹕镇的态度",
                              "episodic_event: 反思/归属感/'一直在找的东西'",
                              "narrative_arc: 第5-6周",
                              "relationship_impression"]))

    cases.append(make_test_case("DAMON_10", "Damon", "best friend",
        "谢谢你一直在这里。", "summer", "Sun", "Beach",
        game_time="20:00", today_actions=["fishing"],
        expected_ltm_context=["persona_seed: 成长弧线",
                              "episodic_event: 感恩/成长/'一直在找的东西'",
                              "narrative_arc: 第5-6周"]))

    # --- Damon 补充 8 条（覆盖更多季节/天气/场景/边界）---
    cases.append(make_test_case("DAMON_11", "Damon", "stranger",
        "嗨。", "fall", "Rain", "Town",
        game_time="17:00", today_actions=["foraging"],
        expected_ltm_context=["persona_seed: 说话风格（简短疏离）",
                              "episodic_event: 无显著匹配（极短输入/陌生人）"]))

    cases.append(make_test_case("DAMON_12", "Damon", "stranger",
        "酒馆里一个人喝酒？", "summer", "Sun", "Saloon",
        game_time="20:00", today_actions=["farming", "fishing"],
        expected_ltm_context=["persona_seed: 工作态度",
                              "episodic_event: 无显著匹配（stranger 阶段事件少）"]))

    cases.append(make_test_case("DAMON_13", "Damon", "acquaintance",
        "图书馆今天人多吗？", "winter", "Sun", "Library",
        game_time="14:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 背景（常去图书馆）",
                              "episodic_event: 图书馆/工作/婉拒帮助"]))

    cases.append(make_test_case("DAMON_14", "Damon", "acquaintance",
        "这个项目看起来挺有意思的。", "spring", "Sun", "Beach",
        game_time="11:00", today_actions=["fishing"],
        expected_ltm_context=["persona_seed: 工作态度",
                              "episodic_event: 海边测绘/保密分享",
                              "preference_belief: seafood（不确定）"]))

    cases.append(make_test_case("DAMON_15", "Damon", "friend",
        "节日的时候你也不休息吗？", "fall", "Sun", "Town",
        game_time="12:00", today_actions=["farming", "foraging"], is_festival=True,
        expected_ltm_context=["persona_seed: 工作态度（强迫症般执着）",
                              "episodic_event: 酒馆/吐槽审批/放松"]))

    cases.append(make_test_case("DAMON_16", "Damon", "friend",
        "生日快乐！来，这是给你的。", "winter", "Sun", "Town",
        game_time="10:00", today_actions=["farming"], is_birthday=True, is_gifting=True,
        expected_ltm_context=["persona_seed: 说话风格",
                              "episodic_event: 送咖啡/雨天",
                              "preference_belief: coffee（like）"]))

    cases.append(make_test_case("DAMON_17", "Damon", "close friend",
        "最近失眠多吗？", "summer", "Rain", "Saloon",
        game_time="19:00", today_actions=["farming", "mining"],
        expected_ltm_context=["persona_seed: 成长弧线/工作焦虑",
                              "episodic_event: 焦虑/抠手指/脆弱",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("DAMON_18", "Damon", "best friend",
        "祖祖城还有什么让你留恋的吗？", "winter", "Rain", "Library",
        game_time="21:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 成长弧线、对鹈鹕镇的态度",
                              "episodic_event: 图书馆/反思祖祖城",
                              "preference_belief: city_life（dislike）",
                              "narrative_arc: 第5-6周",
                              "relationship_impression"]))

    # ========== Abigail（18 条）==========
    cases.append(make_test_case("ABIGAIL_01", "Abigail", "stranger",
        "嘿，你在做什么？", "summer", "Sun", "Town",
        game_time="13:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 性格核心",
                              "episodic_event: first_meeting/作业烦躁"]))

    cases.append(make_test_case("ABIGAIL_02", "Abigail", "acquaintance",
        "听说你喜欢探索矿洞？", "fall", "Rain", "Mountain",
        game_time="16:00", today_actions=["mining", "mining"],
        expected_ltm_context=["persona_seed: 兴趣爱好/矿洞",
                              "episodic_event: 矿洞探险/紫水晶",
                              "preference_belief: mining/amethyst（like）"]))

    cases.append(make_test_case("ABIGAIL_03", "Abigail", "friend",
        "今天有什么有趣的事吗？", "winter", "Sun", "Town",
        game_time="11:00", today_actions=["farming", "foraging"],
        expected_ltm_context=["persona_seed: 家庭关系",
                              "episodic_event: 父母吵架倾诉",
                              "preference_belief: pumpkin（like）"]))

    cases.append(make_test_case("ABIGAIL_04", "Abigail", "friend",
        "湖边下雨的时候真美。", "spring", "Rain", "Mountain",
        game_time="15:00", today_actions=["fishing"],
        expected_ltm_context=["persona_seed: 性格核心",
                              "episodic_event: 雨天/湖边/沉思",
                              "narrative_arc: 第1-2周"]))

    cases.append(make_test_case("ABIGAIL_05", "Abigail", "close friend",
        "你相信超自然的东西吗？", "summer", "Sun", "Beach",
        game_time="21:00", today_actions=["fishing"],
        expected_ltm_context=["persona_seed: 超自然",
                              "episodic_event: 海边/奇怪的光/梦",
                              "episodic_event: 墓地/神秘"]))

    cases.append(make_test_case("ABIGAIL_06", "Abigail", "best friend",
        "你以前觉得鹈鹕镇很无聊，现在呢？", "fall", "Sun", "Mountain",
        game_time="10:00", today_actions=["foraging"],
        expected_ltm_context=["persona_seed: 对鹈鹕镇的态度",
                              "episodic_event: '你这人真有趣'",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("ABIGAIL_07", "Abigail", "dating",
        "我昨晚梦到你了。", "winter", "Sun", "Farm",
        game_time="20:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 浪漫风格",
                              "episodic_event: 做梦/浪漫/亲密",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("ABIGAIL_08", "Abigail", "dating",
        "你想去矿洞冒险吗？", "spring", "Sun", "Mountain",
        game_time="09:00", today_actions=["mining"],
        expected_ltm_context=["persona_seed: 兴趣爱好/矿洞",
                              "episodic_event: 矿洞探险/紫水晶",
                              "preference_belief: mining/amethyst（like）"]))

    cases.append(make_test_case("ABIGAIL_09", "Abigail", "spouse",
        "早安，睡得好吗？", "summer", "Rain", "Farm",
        game_time="07:00", today_actions=[], game_flags=["relationship.married_to.Abigail"],
        expected_ltm_context=["persona_seed: 浪漫风格",
                              "episodic_event: 农场/秋天/平静",
                              "relationship_impression"]))

    cases.append(make_test_case("ABIGAIL_10", "Abigail", "spouse",
        "你觉得人死后会去哪里？", "fall", "Sun", "Farm",
        game_time="22:00", today_actions=["farming"], game_flags=["relationship.married_to.Abigail"],
        expected_ltm_context=["persona_seed: 超自然/神秘",
                              "episodic_event: 墓地/神秘",
                              "relationship_impression"]))

    # --- Abigail 补充 8 条 ---
    cases.append(make_test_case("ABIGAIL_11", "Abigail", "stranger",
        "你没事吧？看起来心情不太好。", "spring", "Rain", "Town",
        game_time="15:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 性格核心/烦躁",
                              "episodic_event: first_meeting/作业烦躁"]))

    cases.append(make_test_case("ABIGAIL_12", "Abigail", "acquaintance",
        "你吹的长笛挺好听的。", "summer", "Sun", "Mountain",
        game_time="14:00", today_actions=["foraging"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/长笛",
                              "episodic_event: 吹长笛/不好意思"]))

    cases.append(make_test_case("ABIGAIL_13", "Abigail", "acquaintance",
        "你也来这里看书？", "winter", "Sun", "Library",
        game_time="16:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 兴趣爱好",
                              "episodic_event: 无显著匹配（acquaintance 阶段事件少）"]))

    cases.append(make_test_case("ABIGAIL_14", "Abigail", "friend",
        "送你一个南瓜！", "fall", "Sun", "Town",
        game_time="11:00", today_actions=["farming"], is_gifting=True,
        expected_ltm_context=["NPC JSON: 兴趣爱好/南瓜",
                              "episodic_event: 南瓜/最喜欢",
                              "preference_belief: pumpkin（like）"]))

    cases.append(make_test_case("ABIGAIL_15", "Abigail", "friend",
        "下雨天来海边散步吗？", "summer", "Rain", "Beach",
        game_time="18:00", today_actions=["fishing"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/冒险",
                              "episodic_event: 雨天/湖边/沉思"]))

    cases.append(make_test_case("ABIGAIL_16", "Abigail", "close friend",
        "你爸妈又吵架了？", "winter", "Rain", "Town",
        game_time="19:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 家庭关系/冲突",
                              "episodic_event: 父母吵架/倾诉",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("ABIGAIL_17", "Abigail", "dating",
        "雨天一起去矿洞？", "fall", "Rain", "Mountain",
        game_time="10:00", today_actions=["mining"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/矿洞",
                              "episodic_event: 矿洞探险/紫水晶",
                              "preference_belief: mining（like）"]))

    cases.append(make_test_case("ABIGAIL_18", "Abigail", "spouse",
        "今天在家做什么好呢？", "winter", "Sun", "Farm",
        game_time="09:00", today_actions=[], game_flags=["relationship.married_to.Abigail"],
        expected_ltm_context=["NPC JSON: 浪漫风格",
                              "episodic_event: 农场/秋天/平静",
                              "relationship_impression"]))

    # ========== Shane（18 条）==========
    cases.append(make_test_case("SHANE_01", "Shane", "stranger",
        "嗨，你是谢恩对吧？", "spring", "Sun", "Town",
        game_time="14:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 性格核心",
                              "episodic_event: first_meeting/粗鲁拒绝"]))

    cases.append(make_test_case("SHANE_02", "Shane", "acquaintance",
        "在 Joja 工作怎么样？", "summer", "Rain", "JojaMart",
        game_time="12:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 背景/工作",
                              "episodic_event: JojaMart/冷淡/上班烦"]))

    cases.append(make_test_case("SHANE_03", "Shane", "friend",
        "来，这杯我请。", "fall", "Sun", "Saloon",
        game_time="20:00", today_actions=["farming", "mining"], is_gifting=True,
        expected_ltm_context=["persona_seed: 酗酒",
                              "episodic_event: 酒馆/请喝酒/'强一点'",
                              "preference_belief: beer（like）"]))

    cases.append(make_test_case("SHANE_04", "Shane", "friend",
        "你还好吗？今天看起来特别消沉。", "winter", "Sun", "Saloon",
        game_time="22:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 抑郁",
                              "episodic_event: 醉酒/脆弱/'活着真累'",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("SHANE_05", "Shane", "close friend",
        "带我去看看你的鸡吧。", "spring", "Sun", "Forest",
        game_time="10:00", today_actions=["farming", "foraging"],
        expected_ltm_context=["persona_seed: 兴趣爱好/鸡",
                              "episodic_event: 鸡/查理/笑了",
                              "preference_belief: chickens（like）"]))

    cases.append(make_test_case("SHANE_06", "Shane", "close friend",
        "昨晚没事吧？我在酒馆看到你了。", "summer", "Rain", "Forest",
        game_time="09:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 抑郁/酗酒",
                              "episodic_event: 陪坐到打烊/信任",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("SHANE_07", "Shane", "dating",
        "今天看起来精神不错。", "fall", "Sun", "Town",
        game_time="11:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 戒酒",
                              "episodic_event: 戒酒/努力/成长",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("SHANE_08", "Shane", "dating",
        "你真的戒酒了？", "winter", "Sun", "Saloon",
        game_time="19:00", today_actions=["mining"],
        expected_ltm_context=["persona_seed: 戒酒/酗酒",
                              "episodic_event: 戒酒/'至少今天没喝'",
                              "preference_belief: beer（like→正在改变）"]))

    cases.append(make_test_case("SHANE_09", "Shane", "spouse",
        "早啊，睡得还好吗？", "spring", "Sun", "Farm",
        game_time="07:00", today_actions=[], game_flags=["relationship.married_to.Shane"],
        expected_ltm_context=["persona_seed: 恋爱风格",
                              "episodic_event: 早上/泡咖啡/关心",
                              "relationship_impression"]))

    cases.append(make_test_case("SHANE_10", "Shane", "spouse",
        "这样的生活是你想要的吗？", "summer", "Rain", "Farm",
        game_time="17:00", today_actions=["farming", "farming"],
        game_flags=["relationship.married_to.Shane"],
        expected_ltm_context=["persona_seed: 自我厌恶/恋爱风格",
                              "episodic_event: 雨天/有你在/感恩",
                              "narrative_arc: 第5-6周",
                              "relationship_impression"]))

    # --- Shane 补充 8 条 ---
    cases.append(make_test_case("SHANE_11", "Shane", "stranger",
        "嘿，你喝了不少啊。", "summer", "Sun", "Saloon",
        game_time="22:00", today_actions=["farming", "fishing"],
        expected_ltm_context=["NPC JSON: 酗酒/性格核心",
                              "episodic_event: first_meeting/粗鲁拒绝"]))

    cases.append(make_test_case("SHANE_12", "Shane", "acquaintance",
        "雨这么大，你还好吗？", "winter", "Rain", "Forest",
        game_time="16:00", today_actions=["foraging"],
        expected_ltm_context=["NPC JSON: 抑郁",
                              "episodic_event: 淋雨/脆弱/'走开'"]))

    cases.append(make_test_case("SHANE_13", "Shane", "acquaintance",
        "下班了？辛苦了。", "fall", "Sun", "Town",
        game_time="18:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 背景/工作",
                              "episodic_event: JojaMart/冷淡/上班烦"]))

    cases.append(make_test_case("SHANE_14", "Shane", "friend",
        "今天想喝点什么？", "spring", "Rain", "Saloon",
        game_time="21:00", today_actions=["farming"], is_gifting=True,
        expected_ltm_context=["NPC JSON: 酗酒",
                              "episodic_event: 酒馆/请喝酒/'强一点'",
                              "preference_belief: beer（like）"]))

    cases.append(make_test_case("SHANE_15", "Shane", "friend",
        "周末去森林走走吧。", "summer", "Sun", "Forest",
        game_time="10:00", today_actions=["foraging"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/鸡",
                              "episodic_event: 鸡/查理/笑了"]))

    cases.append(make_test_case("SHANE_16", "Shane", "close friend",
        "昨晚喝太多了……你没事吧？", "winter", "Rain", "Saloon",
        game_time="21:00", today_actions=["farming", "mining"],
        expected_ltm_context=["NPC JSON: 抑郁/酗酒",
                              "episodic_event: 醉酒/'活着真累'",
                              "episodic_event: 陪坐到打烊/信任",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("SHANE_17", "Shane", "dating",
        "想一起打扫鸡舍吗？", "spring", "Sun", "Forest",
        game_time="11:00", today_actions=["foraging"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/鸡",
                              "episodic_event: 鸡/查理/笑了",
                              "preference_belief: chickens（like）"]))

    cases.append(make_test_case("SHANE_18", "Shane", "spouse",
        "下雨天在家挺好的。", "fall", "Rain", "Farm",
        game_time="15:00", today_actions=["farming"], game_flags=["relationship.married_to.Shane"],
        expected_ltm_context=["NPC JSON: 恋爱风格",
                              "episodic_event: 雨天/有你在/感恩",
                              "relationship_impression"]))

    # ========== Sebastian（18 条）==========
    cases.append(make_test_case("SEBASTIAN_01", "Sebastian", "stranger",
        "你就是住在地下室的那个？", "spring", "Sun", "Mountain",
        game_time="21:00", today_actions=["mining"],
        expected_ltm_context=["persona_seed: 性格核心",
                              "episodic_event: first_meeting/'偏偏选鹈鹕镇'"]))

    cases.append(make_test_case("SEBASTIAN_02", "Sebastian", "acquaintance",
        "这种天气……你在看青蛙吗？", "summer", "Rain", "Mountain",
        game_time="16:00", today_actions=["foraging"],
        expected_ltm_context=["persona_seed: 兴趣爱好",
                              "episodic_event: 雨天/青蛙/自然"]))

    cases.append(make_test_case("SEBASTIAN_03", "Sebastian", "friend",
        "这么晚还出来？", "fall", "Sun", "Mountain",
        game_time="23:00", today_actions=["mining"],
        expected_ltm_context=["persona_seed: 独处/夜晚",
                              "episodic_event: 夜晚/抽烟/独来独往",
                              "preference_belief: rain（like）"]))

    cases.append(make_test_case("SEBASTIAN_04", "Sebastian", "friend",
        "山姆说你台球打得很好。", "winter", "Sun", "Saloon",
        game_time="20:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 兴趣爱好",
                              "episodic_event: 打台球/山姆/放松",
                              "preference_belief: video_games（like）"]))

    cases.append(make_test_case("SEBASTIAN_05", "Sebastian", "close friend",
        "你和玛鲁的关系怎么样？", "spring", "Sun", "Mountain",
        game_time="18:00", today_actions=["mining"],
        expected_ltm_context=["persona_seed: 家庭关系",
                              "episodic_event: 吐槽玛鲁/嫉妒/诚实",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("SEBASTIAN_06", "Sebastian", "close friend",
        "雨天去海滩吧。", "summer", "Rain", "Beach",
        game_time="17:00", today_actions=["fishing"],
        expected_ltm_context=["persona_seed: 兴趣爱好/海滩",
                              "episodic_event: 雨天/海滩/深谈/'动力'",
                              "preference_belief: rain（like）"]))

    cases.append(make_test_case("SEBASTIAN_07", "Sebastian", "best friend",
        "你觉得你在这里交到真正的朋友了吗？", "fall", "Sun", "Beach",
        game_time="20:00", today_actions=["fishing"],
        expected_ltm_context=["persona_seed: 对鹈鹕镇的态度",
                              "episodic_event: '你和山姆是我唯一的朋友'",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("SEBASTIAN_08", "Sebastian", "dating",
        "想我了没？", "winter", "Sun", "Farm",
        game_time="22:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 恋爱风格",
                              "episodic_event: 情话/满身泥/夜晚",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("SEBASTIAN_09", "Sebastian", "spouse",
        "早，又做噩梦了？", "spring", "Sun", "Farm",
        game_time="07:00", today_actions=[], game_flags=["relationship.married_to.Sebastian"],
        expected_ltm_context=["persona_seed: 恋爱风格",
                              "episodic_event: 早上/煮咖啡/噩梦",
                              "relationship_impression"]))

    cases.append(make_test_case("SEBASTIAN_10", "Sebastian", "spouse",
        "今天骑摩托出去兜风吧。", "summer", "Sun", "Farm",
        game_time="14:00", today_actions=["farming"], game_flags=["relationship.married_to.Sebastian"],
        expected_ltm_context=["persona_seed: 兴趣爱好/摩托车、恋爱风格",
                              "episodic_event: 成长/感恩",
                              "preference_belief: motorcycle（不确定）"]))

    # --- Sebastian 补充 8 条 ---
    cases.append(make_test_case("SEBASTIAN_11", "Sebastian", "stranger",
        "下雨天你还出门？", "fall", "Rain", "Mountain",
        game_time="19:00", today_actions=["mining"],
        expected_ltm_context=["NPC JSON: 性格核心/雨天偏好",
                              "episodic_event: first_meeting/'偏偏选鹈鹕镇'"]))

    cases.append(make_test_case("SEBASTIAN_12", "Sebastian", "acquaintance",
        "你在写什么程序？", "spring", "Sun", "Mountain",
        game_time="22:00", today_actions=["mining"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/编程",
                              "episodic_event: 无显著匹配（编程话题事件少）"]))

    cases.append(make_test_case("SEBASTIAN_13", "Sebastian", "acquaintance",
        "你也来酒馆了？", "winter", "Sun", "Saloon",
        game_time="20:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 兴趣爱好",
                              "episodic_event: 无显著匹配（acquaintance 阶段事件少）"]))

    cases.append(make_test_case("SEBASTIAN_14", "Sebastian", "friend",
        "沙滩上散步不错。", "summer", "Sun", "Beach",
        game_time="17:00", today_actions=["fishing"],
        expected_ltm_context=["NPC JSON: 兴趣爱好",
                              "episodic_event: 无显著匹配（海滩非雨天）"]))

    cases.append(make_test_case("SEBASTIAN_15", "Sebastian", "friend",
        "今天雨挺大的。", "spring", "Rain", "Mountain",
        game_time="16:00", today_actions=["foraging"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/雨天",
                              "episodic_event: 雨天/青蛙/自然",
                              "preference_belief: rain（like）"]))

    cases.append(make_test_case("SEBASTIAN_16", "Sebastian", "close friend",
        "搬出来以后感觉怎么样？", "winter", "Sun", "Farm",
        game_time="20:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 对鹈鹕镇的态度",
                              "episodic_event: '走出堡垒'",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("SEBASTIAN_17", "Sebastian", "dating",
        "雨天陪我出去走走？", "spring", "Rain", "Mountain",
        game_time="15:00", today_actions=["mining"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/雨天、恋爱风格",
                              "episodic_event: 雨天/青蛙/自然",
                              "preference_belief: rain（like）"]))

    cases.append(make_test_case("SEBASTIAN_18", "Sebastian", "spouse",
        "今天想做点什么？", "fall", "Rain", "Farm",
        game_time="13:00", today_actions=["farming"], game_flags=["relationship.married_to.Sebastian"],
        expected_ltm_context=["NPC JSON: 恋爱风格",
                              "episodic_event: 成长/感恩/走出堡垒",
                              "relationship_impression"]))

    # ========== Penny（18 条）==========
    cases.append(make_test_case("PENNY_01", "Penny", "stranger",
        "你好，你是潘妮吗？", "spring", "Sun", "Town",
        game_time="10:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 性格核心",
                              "episodic_event: first_meeting/害羞"]))

    cases.append(make_test_case("PENNY_02", "Penny", "acquaintance",
        "今天教孩子们什么了？", "summer", "Sun", "Town",
        game_time="14:00", today_actions=["farming", "foraging"],
        expected_ltm_context=["persona_seed: 兴趣爱好/教育",
                              "episodic_event: 教学/孩子们/热情",
                              "preference_belief: children（like）"]))

    cases.append(make_test_case("PENNY_03", "Penny", "friend",
        "你在看什么书？", "fall", "Rain", "Library",
        game_time="16:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 兴趣爱好/阅读",
                              "episodic_event: 图书馆/安静/逃避",
                              "preference_belief: reading（like）"]))

    cases.append(make_test_case("PENNY_04", "Penny", "friend",
        "你妈妈最近还好吗？", "winter", "Sun", "Town",
        game_time="12:00", today_actions=["farming", "foraging"],
        expected_ltm_context=["persona_seed: 内心挣扎/潘姆",
                              "episodic_event: 妈妈酗酒/担心/倾诉",
                              "narrative_arc: 第3-4周"]))

    cases.append(make_test_case("PENNY_05", "Penny", "close friend",
        "你想带孩子们去哪里郊游？", "spring", "Sun", "Forest",
        game_time="10:00", today_actions=["foraging", "foraging"],
        expected_ltm_context=["persona_seed: 梦想/教育",
                              "episodic_event: 郊游计划/特邀讲师",
                              "preference_belief: children（like）"]))

    cases.append(make_test_case("PENNY_06", "Penny", "best friend",
        "你觉得未来会变得更好吗？", "summer", "Sun", "Town",
        game_time="18:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 梦想/内心挣扎",
                              "episodic_event: 白日梦/花园/农场",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("PENNY_07", "Penny", "dating",
        "和你在一起的时候，时间过得真快。", "fall", "Sun", "Town",
        game_time="16:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 恋爱风格",
                              "episodic_event: 浪漫/'像小说一样精彩'",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("PENNY_08", "Penny", "dating",
        "下雨天……想去图书馆坐坐吗？", "winter", "Rain", "Library",
        game_time="14:00", today_actions=["farming"],
        expected_ltm_context=["persona_seed: 恋爱风格/兴趣爱好",
                              "episodic_event: 图书馆/安静",
                              "preference_belief: reading（like）"]))

    cases.append(make_test_case("PENNY_09", "Penny", "spouse",
        "早安！做了什么好吃的？", "spring", "Sun", "Farm",
        game_time="07:30", today_actions=[], game_flags=["relationship.married_to.Penny"],
        expected_ltm_context=["persona_seed: 恋爱风格",
                              "episodic_event: 早上/做早餐/关心",
                              "relationship_impression"]))

    cases.append(make_test_case("PENNY_10", "Penny", "spouse",
        "还会做以前的噩梦吗？", "summer", "Sun", "Farm",
        game_time="21:00", today_actions=["farming"], game_flags=["relationship.married_to.Penny"],
        expected_ltm_context=["persona_seed: 内心挣扎/噩梦",
                              "episodic_event: 新家/宁静/噩梦消失",
                              "narrative_arc: 第5-6周",
                              "relationship_impression"]))

    # --- Penny 补充 8 条 ---
    cases.append(make_test_case("PENNY_11", "Penny", "stranger",
        "下雨天图书馆人不多吧？", "summer", "Rain", "Library",
        game_time="14:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 性格核心/害羞",
                              "episodic_event: first_meeting/害羞"]))

    cases.append(make_test_case("PENNY_12", "Penny", "acquaintance",
        "孩子们今天学什么？", "fall", "Sun", "Town",
        game_time="11:00", today_actions=["farming", "foraging"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/教育",
                              "episodic_event: 教学/孩子们/热情",
                              "preference_belief: children（like）"]))

    cases.append(make_test_case("PENNY_13", "Penny", "acquaintance",
        "雨天在图书馆看书？", "winter", "Rain", "Library",
        game_time="15:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/阅读",
                              "episodic_event: 图书馆/安静/逃避",
                              "preference_belief: reading（like）"]))

    cases.append(make_test_case("PENNY_14", "Penny", "friend",
        "带孩子们来森林上课吧。", "summer", "Sun", "Forest",
        game_time="10:00", today_actions=["foraging"],
        expected_ltm_context=["NPC JSON: 兴趣爱好/教育",
                              "episodic_event: 郊游计划/特邀讲师",
                              "preference_belief: children（like）"]))

    cases.append(make_test_case("PENNY_15", "Penny", "friend",
        "雨停了，心情好多了。", "spring", "Rain", "Town",
        game_time="13:00", today_actions=["farming", "foraging"],
        expected_ltm_context=["NPC JSON: 内心挣扎",
                              "episodic_event: 图书馆/安静/逃避"]))

    cases.append(make_test_case("PENNY_16", "Penny", "close friend",
        "你心里最大的愿望是什么？", "winter", "Sun", "Library",
        game_time="16:00", today_actions=["farming"],
        expected_ltm_context=["NPC JSON: 梦想/内心挣扎",
                              "episodic_event: 白日梦/花园/农场",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("PENNY_17", "Penny", "dating",
        "去海边走走吧。", "summer", "Sun", "Beach",
        game_time="17:00", today_actions=["fishing"],
        expected_ltm_context=["NPC JSON: 恋爱风格",
                              "episodic_event: '像小说一样精彩'",
                              "narrative_arc: 第5-6周"]))

    cases.append(make_test_case("PENNY_18", "Penny", "spouse",
        "雨天窝在家真好。", "winter", "Rain", "Farm",
        game_time="11:00", today_actions=[], game_flags=["relationship.married_to.Penny"],
        expected_ltm_context=["NPC JSON: 恋爱风格",
                              "episodic_event: 新家/宁静/噩梦消失",
                              "relationship_impression"]))

    return cases


# ============================================================
# 自动映射：expected_ltm_context → expected_memory_ids
# ============================================================

def _resolve_expected_ids(all_ltm: dict, cases: list) -> list:
    """根据 expected_ltm_context 的文本提示，精确匹配 LTM 中的 memory_id。

    匹配策略（逐级降级）：
    1. 精确匹配 topic_tags / topic / week_range / location 索引
    2. 降级：在 content 中做子串搜索
    3. 再无匹配：不添加（宁缺毋滥，避免虚假 recall 拉低指标）
    每个 hint 最多匹配 2 条。
    """

    # 构建关键词→memory_id 索引（同前）
    index = {}
    for npc_id, npc_data in all_ltm.items():
        npc_index = {}
        for mem_type, memories in npc_data.items():
            for mem in memories:
                mid = mem["memory_id"]
                npc_index.setdefault(f"type:{mem_type}", []).append(mid)
                for tag in mem.get("topic_tags", []):
                    npc_index.setdefault(f"tag:{tag}", []).append(mid)
                topic = mem.get("topic", "")
                if topic:
                    npc_index.setdefault(f"topic:{topic}", []).append(mid)
                week = mem.get("week_range", "")
                if week:
                    npc_index.setdefault(f"week:{week}", []).append(mid)
                loc = mem.get("location", "")
                if loc and loc not in ("persona_seed", "narrative"):
                    npc_index.setdefault(f"loc:{loc}", []).append(mid)
                # content 关键词索引（用于降级匹配）
                npc_index.setdefault("_all", []).append({
                    "id": mid,
                    "type": mem_type,
                    "content": mem["content"],
                    "tags": mem.get("topic_tags", []),
                    "topic": topic,
                    "week": week,
                    "loc": loc,
                })
        index[npc_id] = npc_index

    MAX_PER_HINT = 2

    for case in cases:
        npc_id = case["npc_id"]
        npc_index = index.get(npc_id, {})
        all_memories = npc_index.get("_all", [])
        expected_ids = []

        for hint in case.get("expected_ltm_context", []):
            hint = hint.strip()
            if not hint:
                continue

            if ": " in hint:
                prefix, desc = hint.split(": ", 1)
            else:
                prefix = hint
                desc = ""

            # NPC JSON 行不映射
            if prefix == "NPC JSON":
                continue

            matched = set()

            # ---- L1: 精确索引匹配 ----
            type_key = f"type:{prefix}"
            candidates = npc_index.get(type_key, [])

            if desc and desc != "无显著匹配":
                desc_lower = desc.lower()
                kw_candidates = [
                    kw.strip().lower()
                    for kw in desc.replace("（", "(").replace("）", ")").split("/")
                ]
                for kw in kw_candidates:
                    kw_clean = kw.split("(")[0].strip()
                    if not kw_clean:
                        continue
                    # 查 topic_tags
                    for mid in candidates:
                        if mid in npc_index.get(f"tag:{kw_clean}", []):
                            matched.add(mid)
                    # 查 topic
                    for mid in candidates:
                        if mid in npc_index.get(f"topic:{kw_clean}", []):
                            matched.add(mid)
                    # 查 week_range
                    if kw_clean in ("第1-2周", "第3-4周", "第5-6周"):
                        for mid in candidates:
                            if mid in npc_index.get(f"week:{kw_clean}", []):
                                matched.add(mid)
                    # 查 location
                    for mid in candidates:
                        if mid in npc_index.get(f"loc:{kw_clean}", []):
                            matched.add(mid)

            # ---- L2: 降级 → content 子串搜索 ----
            if not matched and desc:
                desc_cleaned = desc.split("（")[0].split("(")[0]  # 去括号说明
                for kw in desc_cleaned.split("/"):
                    kw = kw.strip()
                    if len(kw) < 2:
                        continue
                    for mem in all_memories:
                        if mem["type"] != prefix:
                            continue
                        if kw in mem["content"]:
                            matched.add(mem["id"])
                        elif any(kw in tag for tag in mem["tags"]):
                            matched.add(mem["id"])

            # ---- L3: 无 desc 时，取该 type 下 importance 最高的 1 条 ----
            if not desc and candidates:
                # 找 importance 最高的
                best = None
                best_imp = -1
                for mem in all_memories:
                    if mem["type"] == prefix and mem["id"] in candidates:
                        # 从原始 memory 数据中取 importance（粗略：用 content 长度排序不靠谱，取第一条）
                        if best is None:
                            best = mem["id"]
                if best:
                    matched.add(best)

            # 每个 hint 最多 MAX_PER_HINT 条
            limited = sorted(matched)[:MAX_PER_HINT]
            expected_ids.extend(limited)

        case["expected_memory_ids"] = expected_ids

    return cases


# ============================================================
# 主入口
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ltm = {}
    for gen_func, npc_id in [
        (generate_damon_ltm, "Damon"),
        (generate_abigail_ltm, "Abigail"),
        (generate_shane_ltm, "Shane"),
        (generate_sebastian_ltm, "Sebastian"),
        (generate_penny_ltm, "Penny"),
    ]:
        memories = gen_func()
        by_type = {}
        for mem in memories:
            mt = mem["memory_type"]
            by_type.setdefault(mt, []).append(mem)
        all_ltm[npc_id] = by_type

    with open(OUTPUT_DIR / "ltm_data.json", "w", encoding="utf-8") as f:
        json.dump(all_ltm, f, ensure_ascii=False, indent=2)
    print(f"[OK] LTM data written to ltm_data.json")

    cases = generate_test_cases()

    # 后处理：非 Damon NPC 无 persona_seed，替换 expected_ltm_context 中的 persona_seed 行
    for case in cases:
        if case["npc_id"] != "Damon":
            case["expected_ltm_context"] = [
                ctx.replace("persona_seed: ", "NPC JSON: ")
                for ctx in case["expected_ltm_context"]
            ]

    # 自动映射 expected_memory_ids
    cases = _resolve_expected_ids(all_ltm, cases)

    with open(OUTPUT_DIR / "test_cases.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"[OK] Test cases written to test_cases.jsonl")

    total_ltm = sum(sum(len(m) for m in npc_data.values()) for npc_data in all_ltm.values())
    print(f"\n=== 统计 ===")
    print(f"LTM 记忆总数: {total_ltm}")
    for npc_id, npc_data in all_ltm.items():
        counts = {k: len(v) for k, v in npc_data.items()}
        print(f"  {npc_id}: {counts}")
    print(f"测试用例总数: {len(cases)}")


if __name__ == "__main__":
    main()
