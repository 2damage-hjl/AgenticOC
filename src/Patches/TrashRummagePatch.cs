using HarmonyLib;
using StardewValley;
using Microsoft.Xna.Framework;
using System;
using System.Linq;

namespace AgenticOC.Patches
{
    [HarmonyPatch(typeof(GameLocation), nameof(GameLocation.CheckGarbage))]
    public static class TrashRummagePatch
    {
        static void Postfix(Vector2 tile, Farmer who, GameLocation __instance)
        {
            if (who == null || __instance == null)
                return;

            // Find any NPC within 7 tiles who witnessed the rummaging
            foreach (NPC npc in __instance.characters)
            {
                if (!npc.IsVillager) continue;

                float deltaX = Math.Abs(who.Tile.X - npc.Tile.X);
                float deltaY = Math.Abs(who.Tile.Y - npc.Tile.Y);

                if (deltaX <= 7f && deltaY <= 7f)
                {
                    NpcAttitudeSystem.OnTrashRummagedCaught(npc.Name);
                    NpcTodayActionBuffer.Add(npc.Name, "你抓到玩家在翻垃圾桶，你觉得很脏。");
                }
            }
        }
    }
}
