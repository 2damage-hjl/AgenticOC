import re
import random

GIFT_CATEGORIES = {
    "甜品": {
        "巧克力蛋糕": 220, "粉红蛋糕": 221, "大米布丁": 232, "蓝莓千层酥": 234, 
        "葡萄干布丁": 604, "南瓜派": 608, "蔓越莓糖果": 612
    },
    "水果": {
        "苹果": 613, "杏子": 634, "橙子": 635, "石榴": 637, 
        "樱桃": 638, "菠萝": 832, "芒果": 834
    },
    "饮料": {
        "牛奶": 184, "三倍浓缩咖啡": 253, "果汁": 350, "咖啡": 395, "绿茶": 614
    },
    "漂亮宝石": {
        "绿宝石": 60, "海蓝宝石": 62, "红宝石": 64, "紫水晶": 66, "黄水晶": 68, "翡翠": 70
    },
    "海产品":{
        "金枪鱼": 130, "沙丁鱼": 131, "比目鱼": 267, "龙虾": 715, "蚌": 719, "罗非鱼": 701, "鱿鱼": 151
    },
    "格斯酒馆菜品": {
        "披萨": 206, "豆类火锅": 207, "鱼肉卷": 213, "意大利面": 224, 
        "蔬菜什锦盖饭": 606, "意式蕨菜拌饭": 649
    },
    "增加运势": {
        "幸运午餐": 204, "南瓜汤": 236, "炒鳗鱼": 225, "香蕉布丁": 904, 
        "虾鸡尾酒": 733, "香辣鳗鱼": 226
    },
    "增进钓鱼": {
        "海泡布丁": 265, "海之菜肴": 242, "鳟鱼汤": 219, "龙虾浓汤": 730, 
        "烩鱼汤": 728, "海鲜杂烩汤": 727
    },
    "增进下矿": {
        "秋日恩赐": 235, "塞料面包": 239, "蟹黄糕": 732, "矿工特供": 243, 
        "红梅酱": 238, "墨汁意大利饺": 921, "炒蘑菇": 205
    },
    "增进耕种": {
        "农夫午餐": 240, "完美早餐": 201, "爆炒青椒": 215, "椰汁汤": 218
    },
    "补充生命体力": {
        "肌肉回复药": 351, "生命药水": 773
    },
    "浪漫礼物": {
        "钻石": 72,"精灵珠宝":104, "装饰用扇子": 106, "花束": 458, "珍珠": 797, "财宝箱": 166
    },
    "战斗戒指":{
        "史莱姆克星戒指":520,"战士戒指":521,"吸血戒指":522,"野蛮人戒指":523,
        "窃贼戒指":526,"燃烧弹戒指":811,"保护戒指":861,"吸魂戒指":862,
    },
    "家具":{
        "小型植物":"(F)1362","桌上盆栽":"(F)1363","室内吊篮":"(F)1960","咖啡桌":"(F)724",
        "乡村台灯":"(F)1443","雪地地毯":"(F)1978","大号红色地毯":"(F)2798","墙纸":"(WP)10",
        "墙纸":"(WP)32","墙纸":"(WP)40","墙纸":"(WP)51","墙纸":"(WP)73",
        "墙纸":"(WP)76","墙纸":"(WP)91","墙纸":"(WP)93","墙纸":"(WP)103",
        "复古地板":"(FL)6","黑色地板":"(FL)61","粉粽方格地板":"(FL)48","波点童趣地板":"(FL)24",
    },
    "服装":{
        "水手服":"(S)1019","棕色夹克衫":"(S)1018","浅蓝色条纹衫":"(S)1026","白色背带裤上衣":"(S)1071",
        "领带衬衫":"(S)1123","运动员夹克":"(S)1137","纽扣衬衫":"(S)1140","运动风衣":"(S)1164",
        "船长服":"(S)1239","灰色卫衣":"(S)1160","司机帽":"(H)16","优雅头巾":"(H)64",
        "黑色棒球帽":"(H)DarkBallcap","眼罩":"(H)24","猫耳":"(H)32"
    },
    "古物":{
        "古代玩偶":103,"精灵珠宝":104,"古剑":109,"古代鼓":123,"骨笛":119,"鹦鹉螺化石":586
    }
}

def process_category_gifts(reply: str) -> tuple[str, str, list[dict]]:
    """
    处理礼物类别标签，返回两个版本的文本。
    
    返回:
        - text_with_id:   发给 C# (stdout)，如 "给你这个 [253]"
        - text_with_name: 存入 JSON (记忆)，如 "给你这个 [咖啡]"
        - replacements:   原始数据列表
    """
    
    print(f"\n--- [Gift System] 检测到赠礼指令 ---")
    print(f"原始回复: {reply}")

    replacements = []
    def collect_replacements(match):
        # 使用 .strip() 剔除可能的首尾空格
        category_name = match.group(1).strip()
        
        # 1. 检查品类是否存在
        if category_name in GIFT_CATEGORIES:
            category_items = GIFT_CATEGORIES[category_name]
            
            # 2. 兼容性检查：判断是新版字典还是旧版列表
            if isinstance(category_items, dict):
                item_name = random.choice(list(category_items.keys()))
                item_id = category_items[item_name]
            else:
                # 兼容旧版列表格式
                item_id = random.choice(category_items)
                item_name = "礼物" # 旧版没有名称
            
            gift_info = {"name": item_name, "id": item_id}
            replacements.append(gift_info)
            return f"__REPLACEMENT_{len(replacements)-1}__"
        
        # 3. 匹配失败的调试信息
        print(f"DEBUG: 匹配失败，AI 输出的品类是 '{category_name}'，但字典里只有 {list(GIFT_CATEGORIES.keys())}")
        return ""
    
    # 第一步：扫描并收集
    base_text = re.sub(r'\[GIVE:(.*?)\]', collect_replacements, reply)
    
    # 第二步：双路分发替换
    text_with_id = base_text
    text_with_name = base_text
    
    for i, replacement in enumerate(replacements):
        placeholder = f"__REPLACEMENT_{i}__"
        text_with_id = text_with_id.replace(placeholder, f"[{replacement['id']}]")
        text_with_name = text_with_name.replace(placeholder, f"[{replacement['name']}]")
    
    print(f"结果 A (C#版): {text_with_id}")
    print(f"结果 B (记忆版): {text_with_name}")
    print(f"--- [Gift System] 处理完毕 ---\n")
    
    return text_with_id, text_with_name, replacements