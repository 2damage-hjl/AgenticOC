using System;
using HarmonyLib;
using StardewValley;
using StardewValley.Menus;
using ValleyTalk;
using SObject = StardewValley.Object;

namespace AgenticOC.Patches
{
    /// <summary>
    /// 拦截送礼后的原版对话，改用 AI 生成回复
    /// </summary>
    [HarmonyPatch(typeof(NPC), nameof(NPC.receiveGift))]
    public static class GiftReactionPatch
    {
        // 拦截状态：用于在 Prefix 中标记，在 drawDialogue 补丁中判断
        static bool _suppressGiftDialogue = false;
        static NPC _giftNpc = null;
        static SObject _giftItem = null;
        static GiftReaction _giftReaction = GiftReaction.Neutral;

        /// <summary>
        /// Prefix：在 receiveGift 执行前设置拦截标志，预计算礼物反应
        /// </summary>
        static void Prefix(NPC __instance, SObject o, Farmer giver)
        {
            if (__instance == null || o == null) return;

            _suppressGiftDialogue = true;
            _giftNpc = __instance;
            _giftItem = o;

            // 预计算反应等级
            int taste = __instance.getGiftTasteForThisItem(o);
            _giftReaction = taste switch
            {
                NPC.gift_taste_love => GiftReaction.Love,
                NPC.gift_taste_like => GiftReaction.Like,
                NPC.gift_taste_neutral => GiftReaction.Neutral,
                NPC.gift_taste_dislike => GiftReaction.Dislike,
                NPC.gift_taste_hate => GiftReaction.Hate,
                _ => GiftReaction.Neutral
            };
        }

        /// <summary>
        /// Postfix：原版对话已被拦截，现在发送 AI 请求并显示 AI 回复
        /// </summary>
        static async void Postfix(NPC __instance, SObject o, Farmer giver)
        {
            if (_giftNpc == null) return;

            // 先清除拦截标志，防止影响后续的 drawDialogue 调用
            _suppressGiftDialogue = false;

            try
            {
                // 1. 更新 Attitude 系统
                NpcAttitudeSystem.OnGiftReaction(__instance.Name, _giftReaction);

                // 2. 记录 TodayAction
                string itemName = o.DisplayName;
                string description = _giftReaction switch
                {
                    GiftReaction.Love => $"今天玩家送了你你非常喜欢的 {itemName}。",
                    GiftReaction.Like => $"今天玩家送了你你喜欢的 {itemName}。",
                    GiftReaction.Neutral => $"今天玩家送了你你无感的 {itemName}。",
                    GiftReaction.Dislike => $"今天玩家送了你你不太喜欢的 {itemName}。",
                    GiftReaction.Hate => $"今天玩家送了你你讨厌的 {itemName}。",
                    _ => $"今天玩家送了你 {itemName}。"
                };
                NpcTodayActionBuffer.Add(__instance.Name, description);

                // 3. 构建 AI 请求快照
                var collector = ModEntry.Instance.collector;
                var aiService = ModEntry.Instance.aiService;

                var snapshot = collector.BuildSnapshot(__instance.Name, $"玩家送了你{itemName}", "GIFT");
                snapshot.IsGifting = true;
                snapshot.GiftItem = itemName;
                snapshot.GiftReactionField = _giftReaction.ToString();

                // 4. 显示思考窗口
                Game1.activeClickableMenu = new ThinkingWindow($"{__instance.displayName} 正在思考");

                // 5. 发送 AI 请求
                AIResponse response = await aiService.SendRequestAsync(snapshot);

                // 6. 关闭思考窗口
                Game1.exitActiveMenu();

                // 7. 处理 AI 回复
                if (!string.IsNullOrEmpty(response.Error))
                {
                    ModEntry.Instance.Monitor.Log($"AI Gift Error: {response.Error}", StardewModdingAPI.LogLevel.Error);
                    ShowFallbackDialogue(__instance, _giftReaction);
                }
                else
                {
                    string finalText = response.NpcReply
                        ?.Replace("\r\n", "#$b#")
                        .Replace("\n", "#$b#") ?? "...";

                    __instance.CurrentDialogue.Clear();
                    __instance.CurrentDialogue.Push(new Dialogue(__instance, null, finalText));
                    Game1.drawDialogue(__instance);
                }
            }
            catch (Exception ex)
            {
                ModEntry.Instance.Monitor.Log($"Gift AI Error: {ex.Message}", StardewModdingAPI.LogLevel.Error);
                Game1.exitActiveMenu();
                ShowFallbackDialogue(__instance, _giftReaction);
            }
            finally
            {
                _giftNpc = null;
                _giftItem = null;
            }
        }

        /// <summary>
        /// AI 失败时的兜底回复（简单还原原版风格）
        /// </summary>
        static void ShowFallbackDialogue(NPC npc, GiftReaction reaction)
        {
            string text = reaction switch
            {
                GiftReaction.Love => "这是我最喜欢的东西！太感谢了！",
                GiftReaction.Like => "哇，这礼物真不错，谢谢！",
                GiftReaction.Neutral => "嗯，谢谢你的礼物。",
                GiftReaction.Dislike => "呃……谢谢吧。",
                GiftReaction.Hate => "……你为什么要送我这个？",
                _ => "谢谢。"
            };

            npc.CurrentDialogue.Clear();
            npc.CurrentDialogue.Push(new Dialogue(npc, null, text));
            Game1.drawDialogue(npc);
        }

        /// <summary>
        /// 供 SuppressGiftDialoguePatch 读取拦截状态
        /// </summary>
        public static bool IsSuppressing(NPC npc) => _suppressGiftDialogue && npc == _giftNpc;
    }

    /// <summary>
    /// 拦截原版送礼对话显示：当 GiftReactionPatch 标记了拦截时，跳过 drawDialogue
    /// </summary>
    [HarmonyPatch(typeof(Game1), nameof(Game1.drawDialogue), new[] { typeof(NPC) })]
    public static class SuppressGiftDialoguePatch
    {
        static bool Prefix(NPC speaker)
        {
            // 如果当前正在拦截送礼对话，跳过原版 drawDialogue
            if (GiftReactionPatch.IsSuppressing(speaker))
            {
                return false;
            }
            return true;
        }
    }
}
