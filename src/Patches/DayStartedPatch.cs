using StardewModdingAPI;
using StardewModdingAPI.Events;

namespace Damon.Patches
{
	public class DayStartedPatch
	{
		public static void Register(IModEvents events)
		{
			events.GameLoop.DayStarted += OnDayStarted;
		}

		private static void OnDayStarted(object sender, DayStartedEventArgs e)
		{
			NpcAttitudeSystem.OnDayStarted();
		}
	}
}
