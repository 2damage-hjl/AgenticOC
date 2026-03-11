using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using StardewModdingAPI;

namespace Damon
{
    public class AIService
    {
        private static readonly HttpClient client = new HttpClient();
        // 确保端口和 Python 端一致 (8000)
        private readonly string apiUrl = "http://127.0.0.1:8000/chat";
        private readonly IMonitor monitor;

        public AIService(IMonitor monitor)
        {
            this.monitor = monitor;
            client.Timeout = TimeSpan.FromSeconds(30);
        }

        public async Task<AIResponse> SendRequestAsync(AIRequest snapshot)
        {
            try
            {
                // 1. 序列化
                string jsonPayload = JsonConvert.SerializeObject(snapshot);
                var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

                // 2. 发送请求
                HttpResponseMessage response = await client.PostAsync(apiUrl, content);

                // 3. 检查网络状态
                if (!response.IsSuccessStatusCode)
                {
                    monitor.Log($"Server Error: {response.StatusCode}", LogLevel.Error);
                    return new AIResponse { NpcReply = "...", Error = $"HTTP {response.StatusCode}" };
                }

                // 4. 读取结果
                string jsonResponse = await response.Content.ReadAsStringAsync();

                // 反序列化
                var result = JsonConvert.DeserializeObject<AIResponse>(jsonResponse);
                return result ?? new AIResponse { NpcReply = "...", Error = "Empty Response" };
            }
            catch (Exception ex)
            {
                monitor.Log($"Connection Failed: {ex.Message}", LogLevel.Error);
                // 返回一个友好的错误信息，防止游戏崩溃
                return new AIResponse { NpcReply = "(无法连接到大脑)", Error = ex.Message };
            }
        }
    }
}