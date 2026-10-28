"""
从 citizen.md 解析每个 NPC 对戴蒙和海上宾馆的认知，生成 persona_seed。

citizen.md 格式：
    #NPC_NAME
    记忆行1
    记忆行2

权值规则（按内容类型）：
    直接评价戴蒙性格/为人：0.80-0.85
    表达对戴蒙的好感/信任：0.80
    表达对戴蒙的反感/不信任：0.80
    对戴蒙的行为观察：0.70-0.75
    对戴蒙的随意提及：0.50-0.60
    对宾馆项目强烈支持/反对：0.75
    对宾馆项目一般意见：0.60-0.65
    纯事实陈述/中性观察：0.50-0.55
    带有情感色彩的个人感受：0.70

用法：
    from memory.scripts.citizen_seed import build_all_citizen_seeds, init_all_citizens
    store = MemoryStore()
    init_all_citizens(store)
"""

import uuid
from typing import List, Dict


# ============================================================
# 权值分配辅助
# ============================================================

def _weight(content: str) -> float:
    """根据内容自动判定权值。"""
    # 强情感词 → 高分
    strong_emotion = ["厌烦", "不想要", "不希望", "绝对不行", "不愿意", "责怪",
                      "期待", "很感兴趣", "非常", "太好了", "好孩子", "好",
                      "信任", "欣赏", "喜欢", "享受", "共情", "温柔",
                      "诗意", "诗歌", "被困住的创作者",
                      ]
    # 直接评价戴蒙 → 中高分
    damon_direct = ["戴蒙是", "戴蒙很", "戴蒙有点", "戴蒙看起来", "戴蒙经常",
                    "戴蒙的话", "戴蒙的", "戴蒙挺", "戴蒙不", "戴蒙会",
                    "戴蒙愿意", "戴蒙过于", "我觉得戴蒙",
                    ]
    # 一般意见 → 中分
    general_opinion = ["我觉得", "我认为", "我希望", "我期待", "我比较",
                       "我支持", "我不支持", "我不反对", "我不想",
                       ]
    # 弱意见 → 低分
    weak_opinion = ["听说", "好像", "可能", "或许", "随便", "不太在意",
                    "没什么", "我不太",
                    ]

    base = 0.55  # 默认

    for kw in strong_emotion:
        if kw in content:
            base = max(base, 0.80)
            break
    for kw in damon_direct:
        if kw in content:
            base = max(base, 0.70)
            break
    for kw in general_opinion:
        if kw in content:
            base = max(base, 0.65)
            break
    for kw in weak_opinion:
        if kw in content:
            base = min(base, 0.55)
            break

    # 句子越长、信息越多 → +0.05
    if len(content) > 25:
        base += 0.05
    if len(content) > 40:
        base += 0.05

    return round(min(base, 0.90), 2)


def _make_memory(npc_id: str, content: str) -> dict:
    """构造一条 persona_seed 记忆。"""
    return {
        "memory_id": str(uuid.uuid4()),
        "npc_id": npc_id,
        "content": content,
        "time": "static",
        "location": "persona_seed",
        "importance": _weight(content),
        "memory_type": "persona_seed",
        "status": "active",
        "last_access": 0,
        "source": "citizen_seed",
    }


# ============================================================
# NPC 记忆构建函数（每个 NPC 一个 build 函数）
# ============================================================

def build_Lewis_seed() -> List[Dict]:
    npc = "Lewis"
    items = [
        "鹈鹕镇偶尔会举办一些节日和钓鱼比赛，有许多外地人参加，但都碍于没有居住地无法久留，因此我想建立一个宾馆。",
        "我从以前的老朋友那认识了戴蒙。他从名牌大学毕业，成绩优异，考察后我相信他的业务能力。",
        "深思熟虑以后我决定把宾馆建在海边，景色美丽又能避开生活区。",
        "戴蒙似乎对自己的未来有些迷茫，但我知道他在建筑业上天赋异禀，并且办事认真。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Robin_seed() -> List[Dict]:
    npc = "Robin"
    items = [
        "镇长找我谈过建设宾馆的事情，我持中立意见，但也期待亲手建一个宾馆。",
        "如果真要建，我希望用本地材料。这样维修起来方便，也不会显得像从祖祖城硬搬来的东西。",
        "戴蒙的图纸挺细的，至少不是那种只会画漂亮外观的人，是一个有专业素养的年轻人。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Demetrius_seed() -> List[Dict]:
    npc = "Demetrius"
    items = [
        "我不反对建设，但我希望先看到完整的生态影响报告，尤其是夜间灯光对海鸟迁徙的影响。",
        "戴蒙看起来愿意听取数据，这是好事，至少不是只愿听取预算的人。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Willy_seed() -> List[Dict]:
    npc = "Willy"
    items = [
        "我知道了海上宾馆的事情，游客来了不是坏事，或许有更多人会愿意来钓鱼。",
        "戴蒙的话很少，偶尔会问我一些关于海上宾馆建设的意见。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Linus_seed() -> List[Dict]:
    npc = "Linus"
    items = [
        "我不希望再建设新的宾馆，游客可能会破坏环境，可能有更多人容不下我。",
        "我知道自己人微言轻，宾馆建成后我会把帐篷搭到更隐秘的地方躲起来。",
        "戴蒙看起来不像坏人，但我不喜欢他背后代表的东西。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Pierre_seed() -> List[Dict]:
    npc = "Pierre"
    items = [
        "我觉得宾馆建设非常好，游客多起来，我可以售卖食物和纪念品。",
        "只要不是Joja超市来开连锁宾馆，我觉得都可以谈。",
        "戴蒙看起来是有钱消费的人，可惜不是我的目标客户，没给我带来新消费。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Gus_seed() -> List[Dict]:
    npc = "Gus"
    items = [
        "我觉得宾馆可能不是个坏主意，游客多了酒馆会更热闹。",
        "我期待鹈鹕镇旅游业的发展，希望游客可以享受这里的美食美景和故事。",
        "戴蒙经常来酒馆吃饭，似乎很享受这里的食物，我也享受给他提供美食。",
        "戴蒙的父亲也是做餐饮的，他说之后可能会介绍我们认识，我很期待，希望能学到新美食做法。",
        "戴蒙的话不多，很礼貌，会认真听别人说话。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Penny_seed() -> List[Dict]:
    npc = "Penny"
    items = [
        "我对宾馆的建设没有想法，但担心影响孩子们的生活。",
        "戴蒙的母亲似乎也是老师，我希望可以和她认识。",
        "戴蒙是一个温柔的人，很考虑别人的感受。",
        "戴蒙读过很多书，和他交流教书和学习的经验收获颇多。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Sebastian_seed() -> List[Dict]:
    npc = "Sebastian"
    items = [
        "我听说镇长要建设宾馆，又会多一群人来海边拍照、堵路、问这里有没有网红咖啡，我对此感到厌烦。",
        "戴蒙似乎也不太想和人说话，这点让我觉得很共情。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Abigail_seed() -> List[Dict]:
    npc = "Abigail"
    items = [
        "海上宾馆听起来很酷，也许能见到更多有趣的人，结交志同道合的朋友。",
        "戴蒙有点无趣，成天拿着设计图到处走，过着日复一日的生活。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Shane_seed() -> List[Dict]:
    npc = "Shane"
    items = [
        "听说要搞个什么海上宾馆，随便吧，也不会改变我的生活。",
        "游客多了酒馆可能会变得更拥挤，这对格斯是件好事，但我可能喝不上酒了。",
        "新来的那个建筑师戴蒙有点冷幽默，我不讨厌他。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Sam_seed() -> List[Dict]:
    npc = "Sam"
    items = [
        "听阿比盖尔说镇长要建一个海上宾馆，太好了，那会变得更热闹。",
        "期待游客多了以后可以办更大的演出，或许可以在海边搭个舞台。",
        "戴蒙看起来很严肃，但是他同意给我设计一个海边舞台，我觉得很不错。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Alex_seed() -> List[Dict]:
    npc = "Alex"
    items = [
        "我很支持海上宾馆的设计，也许有更多人来沙滩打排球。",
        "我希望宾馆里能有健身区，并且把这个意见告诉了戴蒙。",
        "戴蒙有点太文绉绉，看起来弱不禁风，我打赌他甚至打不过农场主。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Haley_seed() -> List[Dict]:
    npc = "Haley"
    items = [
        "如果宾馆建的漂亮，我勉强支持，但如果挡住海边的光线，那绝对不行。",
        "对戴蒙好像有点印象，问过我一些问题，我只记得他穿的西装十分考究，但有些死板了。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Elliott_seed() -> List[Dict]:
    npc = "Elliott"
    items = [
        "海上的旅店，潮声入梦，灯火映在窗边……如果设计得当，那会是诗歌愿意停留的地方。",
        "我对戴蒙很感兴趣，我觉得他其实是一个诗意的人，是被现实困住的创作者。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Harvey_seed() -> List[Dict]:
    npc = "Harvey"
    items = [
        "我比较支持海上宾馆的建设，或许能给我带来一些新生意，我不用再跑去周围镇子了。",
        "我和戴蒙还算聊得来，见面会寒暄。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Maru_seed() -> List[Dict]:
    npc = "Maru"
    items = [
        "我对宾馆的建设不太感兴趣，但或许可以给诊所带来一些新生意，哈维应该不那么发愁了。",
        "戴蒙挺认真严谨的，我欣赏他的态度。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Emily_seed() -> List[Dict]:
    npc = "Emily"
    items = [
        "对于海上宾馆，如果那里能让旅行的人感到被欢迎，那会是很好的地方。不过设计里也要留一点让心灵呼吸的空间。",
        "戴蒙的气场有点紧。他需要少看一点图纸，多看看云，或许该教教他瑜伽。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Caroline_seed() -> List[Dict]:
    npc = "Caroline"
    items = [
        "游客多一些应该能给商店带来一些生意，但我私心希望小镇不要变得太吵，尤其是节日以外的时候。",
        "戴蒙很有礼貌，只是看起来总像在赶下一个会议。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Jodi_seed() -> List[Dict]:
    npc = "Jodi"
    items = [
        "如果施工的时候别太吵，我没什么意见。孩子们晚上需要睡觉，Kent也不喜欢太多陌生人。",
        "戴蒙是个认真的孩子，我们没什么交流。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Kent_seed() -> List[Dict]:
    npc = "Kent"
    items = [
        "我不想要建设宾馆，陌生人多了，不一定是好事。尤其是在一个大家都习惯彼此认识的地方。",
        "戴蒙，好像是那个宾馆设计师，我不愿意和他多接近，希望他赶紧离开。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Marnie_seed() -> List[Dict]:
    npc = "Marnie"
    items = [
        "如果游客多了，大家的生意可能都会好一些。不过动物不喜欢太大的噪音，这点可别忘了。",
        "戴蒙看起来很安静，可能不太喜欢吵闹的环境，我觉得他和动物应该相处得不错。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Pam_seed() -> List[Dict]:
    npc = "Pam"
    items = [
        "游客多了，巴士说不定也能多跑几趟。只要他们别在车上吐就行。",
        "游客多了，酒馆的生意应该会更好，格斯的酒值得被更多人品尝到。",
        "戴蒙，有时候会在酒馆见到他，我不太在意，好像和格斯关系不赖。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_George_seed() -> List[Dict]:
    npc = "George"
    items = [
        "宾馆？游客？哼。这个镇已经够吵了，不需要更多人来挡路。",
        "戴蒙？又是一个来折腾小镇的人，希望他赶紧离开。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Evelyn_seed() -> List[Dict]:
    npc = "Evelyn"
    items = [
        "海边以前很安静，孩子们会在那里玩一整个下午。如果要建什么，希望戴蒙能记得这些事。",
        "乔治应该不喜欢小镇变得更吵闹，但是艾利克斯应该希望有更多人能来，或许就有人能陪他打排球了。",
        "戴蒙是个好孩子，会愿意听我讲一些过去的事情。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Clint_seed() -> List[Dict]:
    npc = "Clint"
    items = [
        "我不想建立这个什么海上宾馆，对我的铁匠铺没什么帮助，反而会让小镇变得拥挤。",
        "我跟戴蒙不熟，有点羡慕他的职业形象，想必不缺异性缘。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Wizard_seed() -> List[Dict]:
    npc = "Wizard"
    items = [
        "海与陆地的交界处，本就不是稳定的地方，不要扰动某些边界。",
        "小镇好像来了个外地人，不知道叫什么，希望他能把宾馆选址改到别的地方。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Krobus_seed() -> List[Dict]:
    npc = "Krobus"
    items = [
        "地上的人又要建新的屋子了吗？他们真的很喜欢把空间分成一格一格的。",
        "好像听说从城里来了个建筑师，我还没见过他。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Jas_seed() -> List[Dict]:
    npc = "Jas"
    items = [
        "宾馆会有很大的楼梯吗？我可以从上面跑下来吗？",
        "如果那里有很多陌生人，我就不想去了。",
        "戴蒙是个严肃的大人，好像和潘妮老师聊得不错。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Vincent_seed() -> List[Dict]:
    npc = "Vincent"
    items = [
        "宾馆会有很大的楼梯吗？我可以从上面跑下来吗？",
        "如果那里有很多陌生人，我就不想去了。",
        "戴蒙是个严肃的大人，好像和潘妮老师聊得不错。",
    ]
    return [_make_memory(npc, t) for t in items]


def build_Leah_seed() -> List[Dict]:
    npc = "Leah"
    items = [
        "我不支持海上宾馆的建设，游客多了，生态一定会遭到更大程度的破坏。",
        "本来我责怪戴蒙，但是想到他也只是收钱办事，现在也只是饶他而行，他应该知道我的态度。",
    ]
    return [_make_memory(npc, t) for t in items]


# ============================================================
# NPC 注册表
# ============================================================

ALL_CITIZEN_BUILDERS = {
    "Lewis":       build_Lewis_seed,
    "Robin":       build_Robin_seed,
    "Demetrius":   build_Demetrius_seed,
    "Willy":       build_Willy_seed,
    "Linus":       build_Linus_seed,
    "Pierre":      build_Pierre_seed,
    "Gus":         build_Gus_seed,
    "Penny":       build_Penny_seed,
    "Sebastian":   build_Sebastian_seed,
    "Abigail":     build_Abigail_seed,
    "Shane":       build_Shane_seed,
    "Sam":         build_Sam_seed,
    "Alex":        build_Alex_seed,
    "Haley":       build_Haley_seed,
    "Elliott":     build_Elliott_seed,
    "Harvey":      build_Harvey_seed,
    "Maru":        build_Maru_seed,
    "Emily":       build_Emily_seed,
    "Caroline":    build_Caroline_seed,
    "Jodi":        build_Jodi_seed,
    "Kent":        build_Kent_seed,
    "Marnie":      build_Marnie_seed,
    "Pam":         build_Pam_seed,
    "George":      build_George_seed,
    "Evelyn":      build_Evelyn_seed,
    "Clint":       build_Clint_seed,
    "Wizard":      build_Wizard_seed,
    "Krobus":      build_Krobus_seed,
    "Jas":         build_Jas_seed,
    "Vincent":     build_Vincent_seed,
    "Leah":        build_Leah_seed,
}


# ============================================================
# 批量初始化
# ============================================================

def build_all_citizen_seeds() -> Dict[str, List[Dict]]:
    """生成所有 NPC 的 persona_seed 记忆。

    Returns:
        {npc_id: [memory_dict, ...]}
    """
    result = {}
    for npc_id, build_func in ALL_CITIZEN_BUILDERS.items():
        result[npc_id] = build_func()
    return result


def init_all_citizens(memory_store) -> int:
    """将所有 NPC 的 citizen persona_seed 导入 MemoryStore。

    Args:
        memory_store: MemoryStore 实例

    Returns:
        导入的记忆总数
    """
    from memory.embedded import MemoryStore

    total = 0
    for npc_id, build_func in ALL_CITIZEN_BUILDERS.items():
        # 跳过已存在的
        existing = memory_store.query(
            layer="persona_seed",
            filter={"npc_id": npc_id, "source": "citizen_seed"},
            limit=1,
        )
        if existing and existing.get("ids"):
            print(f"[citizen_seed] {npc_id}: 已存在，跳过")
            continue

        memories = build_func()
        for m in memories:
            memory_store.add(
                "persona_seed",
                m["content"],
                metadata={
                    "npc_id": m["npc_id"],
                    "importance": m["importance"],
                    "time": m["time"],
                    "location": m["location"],
                    "memory_id": m["memory_id"],
                    "memory_type": m["memory_type"],
                    "status": m["status"],
                    "last_access": m["last_access"],
                    "source": m["source"],
                },
                doc_id=m["memory_id"],
            )
            total += 1
        print(f"[citizen_seed] {npc_id}: {len(memories)} 条记忆已导入")

    print(f"\n[citizen_seed] 完成：{len(ALL_CITIZEN_BUILDERS)} 个 NPC，{total} 条记忆")
    return total


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from memory.embedded import MemoryStore

    store = MemoryStore()
    count = init_all_citizens(store)
    print(f"\n总计导入 {count} 条 citizen 记忆")
