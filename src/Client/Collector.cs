using System;
using System.Collections.Generic;
using System.Linq;
using StardewValley;
using StardewModdingAPI;

namespace Damon
{
	public class GameStateCollector
	{
		private readonly IMonitor monitor;

		public GameStateCollector(IMonitor monitor)
		{
			this.monitor = monitor;
		}

		public AIRequest BuildSnapshot(string npcName, string playerInput, string command = "NORMAL")
		{
			var snapshot = new AIRequest
			{
				Command = command,
				NpcName = npcName,
				PlayerInput = playerInput,
				Timestamp = DateTime.Now
			};

			try
			{
				// 1. 基础环境信息
				snapshot.Relationship = GetRelationshipStatus(npcName);
				snapshot.Attitude = NpcAttitudeSystem.GetCurrentAttitude(npcName).ToString();
				snapshot.Weather = GetWeatherString();
				snapshot.Season = Game1.currentSeason;
				snapshot.Location = Game1.player.currentLocation?.Name ?? "unknown";
				snapshot.Year = Game1.year;
				snapshot.DayOfMonth = Game1.dayOfMonth;
				snapshot.GameTime = Game1.getTimeOfDayString(Game1.timeOfDay);
				snapshot.LuckStatus = GetLuckStatus(Game1.player.DailyLuck);

				// --- 2. 玩家状态逻辑 (修改点) ---
				float healthPct = (float)Game1.player.health / Game1.player.maxHealth;
				float staminaPct = Game1.player.Stamina / Game1.player.MaxStamina;

				var statusList = new List<string>();

				// 检查生命值 (< 5%)
				if (healthPct < 0.05f)
					statusList.Add("critical health"); // 对应“低生命值”

				// 检查体力值 (< 5%)
				if (staminaPct < 0.05f)
					statusList.Add("low stamina"); // 对应“低体力值”

				// 生成最终字符串
				if (statusList.Count > 0)
				{
					snapshot.PlayerInfo = string.Join(", ", statusList);
				}
				else
				{
					snapshot.PlayerInfo = "healthy"; // 默认健康
				}

				// 3. 动作
				snapshot.TodayActions = NpcTodayActionBuffer.ConsumeAll(npcName)
					.Select(a => a.Description)
					.ToList();
			}
			catch (Exception ex)
			{
				monitor.Log($"Failed to collect game state: {ex}", LogLevel.Error);
			}

			return snapshot;
		}

		// --- 辅助方法保持不变 ---
		public string GetRelationshipStatus(string npcName)
		{
			if (!Game1.player.friendshipData.TryGetValue(npcName, out Friendship friendship))
				return "stranger";

			int heartLevel = friendship.Points / 250;
			if (friendship.IsMarried()) return "spouse";
			if (friendship.IsEngaged()) return "engaged";
			if (friendship.IsDating()) return "dating";
			if (friendship.IsDivorced()) return "divorced";

			return heartLevel switch
			{
				0 => "stranger",
				1 or 2 => "acquaintance",
				3 or 4 or 5 => "friend",
				6 or 7 or 8 => "close friend",
				9 or 10 => "best friend",
				_ => "stranger"
			};
		}

		public string GetLuckStatus(double dailyLuck)
		{
			if (dailyLuck > 0.05) return "very lucky";
			if (dailyLuck > 0.02) return "lucky";
			if (dailyLuck < -0.05) return "very unlucky";
			if (dailyLuck < -0.02) return "unlucky";
			return "neutral";
		}

		private string GetWeatherString()
		{
			if (Game1.isLightning) return "storm";
			if (Game1.isRaining) return "rainy";
			if (Game1.isSnowing) return "snowy";
			if (Game1.isDebrisWeather) return "windy";
			return "sunny";
		}
	}
}