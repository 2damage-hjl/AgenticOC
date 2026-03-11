using HarmonyLib;
using StardewValley;
using SObject = StardewValley.Object;

namespace Damon.Patches
{
    [HarmonyPatch(typeof(NPC), nameof(NPC.receiveGift))]
    public static class GiftReactionPatch
    {
        static void Postfix(NPC __instance, SObject o, Farmer giver)
        {
            if (__instance == null || o == null)
                return;

            // 如果你想只针对 Damon 生效，可以取消下面这行的注释
            // if (__instance.Name != "Damon") return;

            // 1. 获取原版的好恶判断
            int taste = __instance.getGiftTasteForThisItem(o);

            GiftReaction reaction = taste switch
            {
                NPC.gift_taste_love => GiftReaction.Love,
                NPC.gift_taste_like => GiftReaction.Like,
                NPC.gift_taste_neutral => GiftReaction.Neutral,
                NPC.gift_taste_dislike => GiftReaction.Dislike,
                NPC.gift_taste_hate => GiftReaction.Hate,
                _ => GiftReaction.Neutral
            };

            // 2. 更新 Attitude (情绪/态度) 系统
            // ✅ 必须传入 __instance.Name，确保只修改这个 NPC 的心情
            NpcAttitudeSystem.OnGiftReaction(__instance.Name, reaction);

            // 3. 记录 TodayAction (事实/记忆) 系统
            string itemName = o.DisplayName;

            string description = reaction switch
            {
                GiftReaction.Love =>
                    $"今天玩家送了你你非常喜欢的 {itemName}。",
                GiftReaction.Like =>
                    $"今天玩家送了你你喜欢的 {itemName}。",
                GiftReaction.Neutral =>
                    $"今天玩家送了你你无感的 {itemName}。",
                GiftReaction.Dislike =>
                    $"今天玩家送了你你不太喜欢的 {itemName}。",
                GiftReaction.Hate =>
                    $"今天玩家送了你你讨厌的 {itemName}。",
                _ =>
                    $"今天玩家送了你 {itemName}。"
            };

            NpcTodayActionBuffer.Add(__instance.Name, description);
        }
    }
}