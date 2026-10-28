using System;
using System.Collections.Generic;
using System.Linq;
using StardewValley;
using StardewModdingAPI;

namespace AgenticOC
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
				// 1. ����������Ϣ
				snapshot.Relationship = GetRelationshipStatus(npcName);
				snapshot.Attitude = NpcAttitudeSystem.GetCurrentAttitude(npcName).ToString();
				snapshot.Weather = GetWeatherString();
				snapshot.Season = Game1.currentSeason;
				snapshot.Location = Game1.player.currentLocation?.Name ?? "unknown";
				snapshot.Year = Game1.year;
				snapshot.DayOfMonth = Game1.dayOfMonth;
				snapshot.GameTime = Game1.getTimeOfDayString(Game1.timeOfDay);
				snapshot.LuckStatus = GetLuckStatus(Game1.player.DailyLuck);
				snapshot.IsBirthday = IsNpcBirthday(npcName);
				snapshot.IsFestival = Utility.isFestivalDay(Game1.dayOfMonth, (Season)System.Enum.Parse(typeof(Season), Game1.currentSeason, true));
				snapshot.Route = GetRouteStatus();
				snapshot.GameFlags = CollectGameFlags(npcName);

				// --- 2. ���״̬�߼� (�޸ĵ�) ---
				float healthPct = (float)Game1.player.health / Game1.player.maxHealth;
				float staminaPct = Game1.player.Stamina / Game1.player.MaxStamina;

				var statusList = new List<string>();

				// �������ֵ (< 5%)
				if (healthPct < 0.05f)
					statusList.Add("critical health"); // ��Ӧ��������ֵ��

				// �������ֵ (< 5%)
				if (staminaPct < 0.05f)
					statusList.Add("low stamina"); // ��Ӧ��������ֵ��

				// ���������ַ���
				if (statusList.Count > 0)
				{
					snapshot.PlayerInfo = string.Join(", ", statusList);
				}
				else
				{
					snapshot.PlayerInfo = "healthy"; // Ĭ�Ͻ���
				}

				// 3. ����
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

		// --- �����������ֲ��� ---
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

		private bool IsNpcBirthday(string npcName)
		{
			NPC npc = Game1.getCharacterFromName(npcName);
			if (npc == null) return false;
			return npc.Birthday_Season == Game1.currentSeason && npc.Birthday_Day == Game1.dayOfMonth;
		}

		private string GetRouteStatus()
		{
			// �������������
			if (Game1.player.mailReceived.Contains("ccIsComplete"))
				return "community_center_completed";

			// Joja ·��
			if (Game1.player.mailReceived.Contains("JojaMember"))
			{
				// ��� Joja ȫ����Ŀ�Ƿ���ɣ�5 ������
				string[] jojaProjects = { "ccJojaBoilerRoom", "ccJojaCraftsRoom", "ccJojaVault", "ccJojaPantry", "ccJojaFishTank" };
				foreach (var project in jojaProjects)
				{
					if (!Game1.player.mailReceived.Contains(project))
						return "joja_active";
				}
				return "joja_completed";
			}

			return "pre_choice";
		}

		private List<string> CollectGameFlags(string npcName)
		{
			var flags = new List<string>();

			// relationship.married_to.{NPC��}
			if (Game1.player.friendshipData.TryGetValue(npcName, out Friendship friendship))
			{
				if (friendship.IsMarried())
					flags.Add($"relationship.married_to.{npcName}");
			}

			// world.ginger_island.beach_resort.opened
			if (Game1.player.mailReceived.Contains("islandResort"))
				flags.Add("world.ginger_island.beach_resort.opened");

			return flags;
		}
	}
}