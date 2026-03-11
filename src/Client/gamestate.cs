using System.Collections.Generic;
using Newtonsoft.Json;

namespace Damon
{
    public class AIRequest
    {
        // --- 基础控制字段 ---
        [JsonProperty("command")]
        public string Command { get; set; } = "NORMAL";

        [JsonProperty("npc_id")]
        public string NpcName { get; set; } = "Damon";

        [JsonProperty("player_input")]
        public string PlayerInput { get; set; } = "";

        // --- 环境信息 ---
        [JsonProperty("relationship")]
        public string Relationship { get; set; } = "stranger";

        [JsonProperty("attitude")]
        public string Attitude { get; set; } = "Neutral";

        [JsonProperty("today_actions")]
        public List<string> TodayActions { get; set; } = new List<string>();

        [JsonProperty("weather")]
        public string Weather { get; set; } = "sunny";

        [JsonProperty("season")]
        public string Season { get; set; } = "spring";

        [JsonProperty("location")]
        public string Location { get; set; } = "unknown";

        [JsonProperty("year")]
        public int Year { get; set; } = 1;

        [JsonProperty("dayOfMonth")]
        public int DayOfMonth { get; set; } = 1;

        [JsonProperty("game_time")]
        public string GameTime { get; set; } = "0600";

        [JsonProperty("luckystatus")]
        public string LuckStatus { get; set; } = "neutral";

        // --- 玩家状态 (修改后) ---
        // 删除了具体的数值 int 字段
        [JsonProperty("player_info")]
        public string PlayerInfo { get; set; } = "healthy";

        [JsonProperty("extra")]
        public Dictionary<string, object> Extra { get; set; } = new Dictionary<string, object>();

        [JsonIgnore]
        public System.DateTime Timestamp { get; set; } = System.DateTime.Now;
    }

    public class AIResponse
    {
        [JsonProperty("npc_reply")]
        public string NpcReply { get; set; } = "...";

        [JsonProperty("command")]
        public string Command { get; set; } = "NORMAL";

        [JsonProperty("error")]
        public string? Error { get; set; } = null;

        // ✅ 新增：多人批量模式用这个
        [JsonProperty("batch_data")]
        public Dictionary<string, int>? BatchData { get; set; }
    }
}