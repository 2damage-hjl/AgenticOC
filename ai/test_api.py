import requests
import time

# 服务器地址
BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """测试服务器存活状态"""
    print("\n[1] 正在测试健康检查接口 (Health Check)...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ 服务器正常在线:", response.json())
            return True
        else:
            print("❌ 服务器返回异常状态码:", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器。请确保运行了 'python server.py'")
        return False

def test_chat_normal():
    """测试正常的对话流程"""
    print("\n[2] 正在测试对话接口 (Chat - Normal)...")
    
    # 模拟 C# 发送的数据 (对应 GameStateInput)
    payload = {
        "command": "NORMAL",
        "npc_id": "Abigail", # 测试跟阿比盖尔聊天
        "player_input": "嘿，不管今天天气如何，看到你我就觉得心情不错。",
        
        # 游戏时间与日期
        "year": 2,
        "season": "fall",
        "dayOfMonth": 15,
        "game_time": "", # 这个可以让服务器自己算，留空没事
        
        # 环境信息
        "location": "Town",
        "relationship": "friend",
        "attitude": "friendly",
        "weather": "Rain",
        
        # 玩家状态
        "player_info": "healthy",
        "luckystatus": "Good",
        "today_actions": ["gave_gift_to_Abigail", "fished_in_river"],
        
        # 额外信息
        "extra": {
            "debug_mode": True
        }
    }

    try:
        start_time = time.time()
        # 发送 POST 请求
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功! (耗时: {end_time - start_time:.2f}s)")
            print("--------------------------------------------------")
            print(f"🤖 NPC回复: {result.get('npc_reply')}")
            print(f"📜 指令状态: {result.get('command')}")
            if result.get('error'):
                print(f"⚠️ 内部警告: {result.get('error')}")
            print("--------------------------------------------------")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print("详细错误:", response.text)

    except Exception as e:
        print(f"❌ 发送请求时出错: {e}")

def test_chat_trigger_error():
    """(可选) 测试报错情况，看服务器会不会崩"""
    print("\n[3] 正在测试容错性 (发送非法数据)...")
    # 故意少传参数，或者传错类型
    payload = {
        "command": "NORMAL",
        # 缺少 npc_id, 缺少必要字段
        "player_input": 12345 # 故意传数字
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    if response.status_code == 422: # 422 Unprocessable Entity 是 FastAPI 对类型错误的默认返回
        print("✅ 容错测试通过: 服务器正确拦截了非法数据 (返回 422)。")
    elif response.status_code == 500:
        print("⚠️ 服务器崩溃了 (返回 500)。")
    else:
        print(f"❓ 返回状态码: {response.status_code}")

if __name__ == "__main__":
    # 1. 先测健康
    if test_health():
        # 2. 再测对话
        test_chat_normal()
        # 3. 测容错
        test_chat_trigger_error()
    else:
        print("\n🚫 测试终止：无法连接服务器。")