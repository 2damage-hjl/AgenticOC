using System;
using Microsoft.Xna.Framework;
using StardewModdingAPI;
using StardewValley;

namespace AgenticOC
{
    /// <summary>Generic NPC utility for finding and interacting with villagers</summary>
    public class NPCFinder
    {
        private readonly IMonitor monitor;

        public NPCFinder(IMonitor monitor)
        {
            this.monitor = monitor;
        }

        /// <summary>Find an NPC by name across all locations</summary>
        public NPC? FindByName(string npcName)
        {
            try
            {
                foreach (GameLocation location in Game1.locations)
                {
                    foreach (NPC npc in location.characters)
                    {
                        if (npc.Name.Equals(npcName, StringComparison.OrdinalIgnoreCase))
                        {
                            return npc;
                        }
                    }
                }

                monitor.Log($"NPC '{npcName}' not found", LogLevel.Warn);
                return null;
            }
            catch (Exception ex)
            {
                monitor.Log($"Error finding NPC '{npcName}': {ex.Message}", LogLevel.Error);
                return null;
            }
        }
    }
}