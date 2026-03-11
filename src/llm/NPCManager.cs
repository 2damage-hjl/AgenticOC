using System;
using Microsoft.Xna.Framework; // 添加这个
using StardewModdingAPI;
using StardewValley;

namespace Damon
{
    /// <summary>管理Damon NPC的状态和行为</summary>
    public class NPCManager
    {
        private readonly IMonitor monitor;
        private NPC damon;

        public NPC Damon => damon;
        public bool IsDamonFound => damon != null;

        public NPCManager(IMonitor monitor)
        {
            this.monitor = monitor;
        }

        /// <summary>查找Damon NPC</summary>
        public bool FindDamonNPC()
        {
            try
            {
                // 先重置
                damon = null;

                // 在所有地点查找名为"Damon"的NPC
                foreach (GameLocation location in Game1.locations)
                {
                    foreach (NPC npc in location.characters)
                    {
                        if (npc.Name.Equals("Damon", StringComparison.OrdinalIgnoreCase))
                        {
                            damon = npc;
                            //monitor.Log($"✅ 找到Damon在: {location.Name} (位置: {npc.Tile})", LogLevel.Debug); 
                            return true;
                        }
                    }
                }

                monitor.Log("❌ 未找到Damon角色", LogLevel.Warn);
                return false;
            }
            catch (Exception ex)
            {
                monitor.Log($"💥 查找Damon时出错: {ex.Message}", LogLevel.Error);
                return false;
            }
        }

        /// <summary>检查玩家是否靠近Damon</summary>
        public bool IsPlayerNearDamon()
        {
            if (damon == null || !Context.IsWorldReady)
            {
                return false;
            }

            // 检查是否在同一地点
            if (damon.currentLocation != Game1.player.currentLocation)
            {
                //monitor.Log($"📍 玩家和Damon不在同一地点: 玩家在{Game1.player.currentLocation.Name}, Damon在{damon.currentLocation?.Name}", LogLevel.Trace);
                return false;
            }

            // 计算玩家与Damon之间的距离
            Vector2 playerTile = Game1.player.Tile; // 修复：使用 Tile 而不是 getTileLocation()
            Vector2 damonTile = damon.Tile; // 修复：使用 Tile 而不是 getTileLocation()
            float distance = Vector2.Distance(playerTile, damonTile);

            bool isNear = distance <= 3f;

            // 记录距离信息用于调试
            if (isNear)
            {
                //monitor.Log($"📏 玩家靠近Damon - 距离: {distance:F1}格 (玩家: {playerTile}, Damon: {damonTile})", LogLevel.Trace);
            }

            return isNear;
        }

        /// <summary>获取玩家与Damon的精确距离</summary>
        public float GetDistanceToDamon()
        {
            if (damon == null || damon.currentLocation != Game1.player.currentLocation)
                return float.MaxValue;

            return Vector2.Distance(Game1.player.Tile, damon.Tile); // 修复：使用 Tile 而不是 getTileLocation()
        }

        /// <summary>让Damon面向玩家</summary>
        public void FacePlayer()
        {
            if (damon == null) return;

            try
            {
                Vector2 playerTile = Game1.player.Tile; // 修复：使用 Tile 而不是 getTileLocation()
                Vector2 damonTile = damon.Tile; // 修复：使用 Tile 而不是 getTileLocation()

                if (playerTile.Y < damonTile.Y) // 玩家在Damon上方
                    damon.faceDirection(2);
                else if (playerTile.Y > damonTile.Y) // 玩家在Damon下方
                    damon.faceDirection(0);
                else if (playerTile.X < damonTile.X) // 玩家在Damon左侧
                    damon.faceDirection(1);
                else if (playerTile.X > damonTile.X) // 玩家在Damon右侧
                    damon.faceDirection(3);

                //monitor.Log($"👀 Damon面向玩家 (方向: {damon.FacingDirection})", LogLevel.Trace);
            }
            catch (Exception ex)
            {
                monitor.Log($"💥 设置Damon面向时出错: {ex.Message}", LogLevel.Debug);
            }
        }

        /// <summary>检查Damon是否仍然有效</summary>
        public bool ValidateDamon()
        {
            if (damon == null)
                return false;

            // 检查NPC是否仍然存在于游戏中
            bool exists = false;
            foreach (GameLocation location in Game1.locations)
            {
                if (location.characters.Contains(damon))
                {
                    exists = true;
                    break;
                }
            }

            if (!exists)
            {
                monitor.Log("⚠️ Damon角色已从游戏中移除，需要重新查找", LogLevel.Warn);
                damon = null;
            }

            return exists;
        }
    }
}