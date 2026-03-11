using System;
using StardewModdingAPI;
using StardewModdingAPI.Events;
using StardewValley;
using HarmonyLib;
using Microsoft.Xna.Framework;
using static StardewValley.Dialogue;
using ValleyTalk;
using ValleyTalk.UI;

namespace Damon
{
    public class ModEntry : Mod
    {
        // 使用 default! 消除 "未初始化" 警告，因为我们在 Entry 中初始化了
        private GameStateCollector collector = default!;
        private AIService aiService = default!;
        private bool isQKeySuppressed = false;

        // --- 新增：对话状态控制 ---
        private int currentTurnCount = 0;
        private const int MAX_TURNS = 10;
        private NPC? currentTalkingNpc = null;

        public override void Entry(IModHelper helper)
        {
            var harmony = new Harmony(this.ModManifest.UniqueID);
            harmony.PatchAll();

            this.collector = new GameStateCollector(Monitor);
            this.aiService = new AIService(Monitor);

            helper.Events.GameLoop.DayStarted += OnDayStarted;
            helper.Events.Input.ButtonPressed += OnButtonPressed;
            helper.Events.Input.ButtonReleased += OnButtonReleased;

            Monitor.Log("AI Mod Loaded. Press Q near an NPC to chat.", LogLevel.Info);
        }

        // 修复：参数类型加上 '?' 以匹配 EventHandler 签名
        private async void OnDayStarted(object? sender, DayStartedEventArgs e)
        {
            NpcAttitudeSystem.OnDayStarted();

            // 1. 获取所有昨天发生了互动的 NPC 列表
            var activeNpcs = NpcTodayActionBuffer.GetAllActiveNpcNames();

            Monitor.Log($"[System] Starting daily summary for {activeNpcs.Count} NPCs...", LogLevel.Info);

            // 2. 遍历每一个 NPC 发送请求
            foreach (var npcName in activeNpcs)
            {
                // 构建该 NPC 的快照
                var snapshot = collector.BuildSnapshot(npcName, "", "END_DAY");

                try
                {
                    // 这里不需要 await（不需要等一个完了再发下一个），可以并行发送
                    // 但为了防止服务器瞬间压力过大，简单的 await 循环是最稳妥的
                    await aiService.SendRequestAsync(snapshot);
                    Monitor.Log($"[System] Sent END_DAY for {npcName}", LogLevel.Trace);
                }
                catch (Exception ex)
                {
                    Monitor.Log($"[System] Failed to summarize {npcName}: {ex.Message}", LogLevel.Warn);
                }
            }

            // 3. 所有请求发完后，清空 Buffer
            NpcTodayActionBuffer.ClearAtDayEnd();

            // 每周周日触发 (Day 7, 14, 21...)
            if (Game1.dayOfMonth % 7 == 0)
            {
                Monitor.Log("[System] Triggering Weekly Batch Summary...", LogLevel.Info);

                // 发送通用指令，不需要指定 npc_id (传 "System" 或空都行)
                var weeklySnapshot = collector.BuildSnapshot("System", "", "WEEKLY_SUMMARY");

                var response = await aiService.SendRequestAsync(weeklySnapshot);

                // ✅ 处理批量更新指令
                if (response.Command == "BATCH_UPDATE_FRIENDSHIP" && response.BatchData != null)
                {
                    foreach (var kvp in response.BatchData)
                    {
                        string npcName = kvp.Key;
                        int delta = kvp.Value;

                        // 找到对应的 NPC 对象
                        NPC target = Game1.getCharacterFromName(npcName);
                        if (target != null)
                        {
                            Game1.player.changeFriendship(delta, target);
                            Monitor.Log($"[Weekly] {npcName} friendship changed by {delta}", LogLevel.Info);
                        }
                        else
                        {
                            Monitor.Log($"[Weekly] Could not find NPC named {npcName}", LogLevel.Warn);
                        }
                    }

                    Game1.chatBox.addMessage("System: Weekly memories consolidated.", Color.Gold);
                }
            }
        }

        private void OnButtonReleased(object? sender, ButtonReleasedEventArgs e)
        {
            if (e.Button == SButton.Q) isQKeySuppressed = false;
        }

        private async void OnButtonPressed(object? sender, ButtonPressedEventArgs e)
        {
            if (!Context.IsWorldReady || e.Button != SButton.Q || isQKeySuppressed) return;

            Helper.Input.Suppress(e.Button);
            isQKeySuppressed = true;

            // 寻找最近的 NPC
            NPC? targetNpc = FindNearestNpc(radiusTiles: 3);

            // 修复：使用 IsVillager (属性) 而不是 isVillager() (方法)
            // 修复：targetNpc 可能是 null
            if (targetNpc == null || !targetNpc.IsVillager) return;

            // 1. 收集状态
            var snapshot = collector.BuildSnapshot(targetNpc.Name, playerInput: "", command: "NORMAL");

            // 2. 打开思考窗口
            // 修复：使用 displayName (小写 d) 属性
            Game1.activeClickableMenu = new ThinkingWindow($"{targetNpc.displayName} 正在思考");

            try
            {
                // 3. 发送请求
                AIResponse response = await aiService.SendRequestAsync(snapshot);

                // 4. 关闭思考窗口
                Game1.exitActiveMenu();

                // 5. 处理结果
                if (!string.IsNullOrEmpty(response.Error))
                {
                    Monitor.Log($"AI Error: {response.Error}", LogLevel.Error);
                    Game1.drawObjectDialogue($"(AI Error: {response.Error})");
                }
                else
                {
                    Monitor.Log($"[{targetNpc.Name}] 回复: {response.NpcReply}", LogLevel.Info);

                    var text = response.NpcReply;

                    if (string.IsNullOrWhiteSpace(text))
                        text = "...（他似乎在思考）";

                    // 使用正确构造器
                    var dialogue = new Dialogue(targetNpc, null, text);

                    targetNpc.CurrentDialogue.Clear();
                    targetNpc.CurrentDialogue.Push(dialogue);
                    Game1.drawDialogue(targetNpc);

                    Game1.afterDialogues = () =>
                    {
                        // 气泡关闭后，立即显示选项
                        ShowContinueOptions(targetNpc);
                    };
                }
            }
            catch (Exception ex)
            {
                Monitor.Log($"Critical Error: {ex.Message}", LogLevel.Error);
                // 确保出错时关闭窗口
                Game1.exitActiveMenu();
            }
        }

        /// <summary>
        /// 核心处理函数：发送请求 -> AI回复 -> 显示气泡 -> 设置下一步动作
        /// </summary>
        private async System.Threading.Tasks.Task ProcessConversationTurn(NPC npc, string playerInput, string command, bool isGreeting)
        {
            // 构建快照
            var snapshot = collector.BuildSnapshot(npc.Name, playerInput, command);

            // 显示思考中
            Game1.activeClickableMenu = new ThinkingWindow($"{npc.displayName} 正在思考");

            try
            {
                // 发送请求
                AIResponse response = await aiService.SendRequestAsync(snapshot);

                // 收到回复后，立刻关闭思考窗口，防止遮挡
                Game1.exitActiveMenu();

                if (!string.IsNullOrEmpty(response.Error))
                {
                    Monitor.Log($"AI Error: {response.Error}", LogLevel.Error);
                    Game1.drawObjectDialogue($"(连接错误: {response.Error})");
                    return;
                }

                // 仅仅做格式转换：把换行符变成游戏的分段符，不做任何其他清洗
                // 如果不做这个，AI输出多行文本时游戏会显示异常
                string finalDialogueText = response.NpcReply
                    ?.Replace("\r\n", "#$b#")
                    .Replace("\n", "#$b#") ?? "...";

                Monitor.Log($"[ProcessTurn] AI回复: {finalDialogueText}", LogLevel.Info);

                // ==============================================================
                // ✅ 修复：使用 Dialogue 构造函数，彻底解决 "Unable to parse string path"
                // ==============================================================

                // 1. 清空当前对话栈
                npc.CurrentDialogue.Clear();

                // 2. 构造对话对象 (第二个参数为 null 表示直接使用文本，不是 Key)
                var dialogue = new Dialogue(npc, null, finalDialogueText);

                // 3. 压入堆栈
                npc.CurrentDialogue.Push(dialogue);

                // 4. 显示
                Game1.drawDialogue(npc);

                // ==============================================================
                // 🔑 状态流转：气泡关闭后的回调
                // ==============================================================
                Game1.afterDialogues = () =>
                {
                    // 只有在非结束指令时才显示后续选项
                    if (command != "END_DIALOGUE" && command != "CLEAR_DIALOGUE")
                    {
                        ShowContinueOptions(npc);
                    }
                };
            }
            catch (Exception ex)
            {
                Monitor.Log($"Critical Error in ProcessTurn: {ex.Message}", LogLevel.Error);
                // 确保出错也能关掉窗口
                Game1.exitActiveMenu();
            }
        }

        /// <summary>
        /// 显示“继续对话 / 保持沉默”的选项
        /// </summary>
        private void ShowContinueOptions(NPC npc)
        {
            // 检查轮次限制
            if (currentTurnCount >= MAX_TURNS)
            {
                Monitor.Log("已达到最大对话轮次，强制结束。", LogLevel.Info);
                // ✅ 重置计数器
                currentTurnCount = 0;
                // 发送结束指令（后台发送，不等待）
                _ = aiService.SendRequestAsync(collector.BuildSnapshot(npc.Name, "", "END_DIALOGUE"));
                return;
            }

            // 定义选项
            Response[] responses = {
                new Response("Continue", "继续对话"),
                new Response("Silent", "保持沉默")
            };

            // 创建原生提问框
            Game1.currentLocation.createQuestionDialogue(
                $"({currentTurnCount}/{MAX_TURNS}) 你想要...",
                responses,
                (Farmer who, string answerKey) => OnQuestionAnswered(who, answerKey, npc)
            );
        }

        /// <summary>
        /// 处理原生选项的选择结果
        /// </summary>
        private void OnQuestionAnswered(Farmer who, string answerKey, NPC npc)
        {
            if (answerKey == "Silent")
            {
                // === 选项 A: 保持沉默 ===
                Monitor.Log("玩家选择沉默，结束对话。", LogLevel.Info);
                // ✅ 重置计数器
                currentTurnCount = 0;
                _ = aiService.SendRequestAsync(collector.BuildSnapshot(npc.Name, "", "END_DIALOGUE"));
                // 此时对话自然结束，没有任何 UI
            }
            else if (answerKey == "Continue")
            {
                // === 选项 B: 继续对话 ===
                // 打开我们自定义的输入 UI
                OpenCustomInputMenu(npc);
            }
        }

        /// <summary>
        /// 打开自定义输入 UI 并绑定事件
        /// </summary>
        private void OpenCustomInputMenu(NPC npc)
        {
            var chatMenu = new DialogueTextInputMenu($"与 {npc.displayName} 对话中...");

            // --- 绑定：提交 (OK) ---
            chatMenu.OnSubmit += async (text) =>
            {
                currentTurnCount++; // 轮次 +1
                // 递归调用处理函数，发送 NORMAL 指令
                await ProcessConversationTurn(npc, text, "NORMAL", isGreeting: false);
            };

            // --- 绑定：取消 (Cancel) ---
            chatMenu.OnCancel += () =>
            {
                Monitor.Log("在输入框按下了 Cancel，结束对话。", LogLevel.Info);
                // ✅ 重置计数器
                currentTurnCount = 0;
                _ = aiService.SendRequestAsync(collector.BuildSnapshot(npc.Name, "", "END_DIALOGUE"));
            };

            // --- 绑定：清空/清除 (Clear) ---
            chatMenu.OnClearRequested += (clearAll) =>
            {
                Monitor.Log("在输入框按下了 Clear，清除记忆并结束。", LogLevel.Warn);
                // ✅ 重置计数器
                currentTurnCount = 0;
                _ = aiService.SendRequestAsync(collector.BuildSnapshot(npc.Name, "", "CLEAR_DIALOGUE"));
                // 这里需要手动关闭菜单，因为 OnClearRequested 只是通知
                Game1.exitActiveMenu();
            };

            Game1.activeClickableMenu = chatMenu;
        }

        private NPC? FindNearestNpc(int radiusTiles)
        {
            if (Game1.player?.currentLocation == null) return null;
            Vector2 playerPos = Game1.player.Tile;
            NPC? nearest = null;
            double minDistance = double.MaxValue;

            foreach (NPC npc in Game1.player.currentLocation.characters)
            {
                Vector2 npcPos = npc.Tile;
                double distance = Vector2.Distance(playerPos, npcPos);
                if (distance <= radiusTiles && distance < minDistance)
                {
                    minDistance = distance;
                    nearest = npc;
                }
            }
            return nearest;
        }
    }
}
