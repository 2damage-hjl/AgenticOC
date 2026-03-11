using HarmonyLib;
using StardewValley;
using Microsoft.Xna.Framework;
using System;
using System.Linq;

namespace Damon.Patches
{
    [HarmonyPatch(typeof(GameLocation), nameof(GameLocation.CheckGarbage))]
    public static class DamonTrashRummagePatch
    {
        static void Postfix(Vector2 tile, Farmer who, GameLocation __instance)
        {
            if (who == null || __instance == null)
                return;

            // 找 Damon
            var damon = __instance.characters.FirstOrDefault(n => n.Name == "Damon");
            if (damon == null)
                return;

            // 计算玩家和 Damon 的距离（格子）
            float deltaX = Math.Abs(who.Tile.X - damon.Tile.X);
            float deltaY = Math.Abs(who.Tile.Y - damon.Tile.Y);

            // 7×7 范围内才触发
            if (deltaX <= 7f && deltaY <= 7f)
            {
                // 调用自定义逻辑
                Damon.NpcAttitudeSystem.OnTrashRummagedCaught(__instance.Name);
                NpcTodayActionBuffer.Add(__instance.Name, "你抓到玩家在翻垃圾桶，你觉得很脏。");
            }
        }
    }
}
