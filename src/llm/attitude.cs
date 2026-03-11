using System;
using System.Collections.Generic;
using System.Linq;

namespace Damon
{
    /// <summary>
    /// NPC 当下交流姿态（非关系，指当下的心情/态度）
    /// </summary>
    public enum NpcAttitude
    {
        Angry,       // 明显生气
        Irritated,   // 不耐烦
        Neutral,     // 情绪中性
        Pleasant,    // 好说话
        Warm         // 情绪温和
    }

    /// <summary>
    /// 星露谷送礼反馈枚举
    /// </summary>
    public enum GiftReaction
    {
        Love,
        Like,
        Neutral,
        Dislike,
        Hate
    }

    public class TodayAction
    {
        public string Description;   // 客观事实
        public bool Consumed;        // 是否已喂给 AI
    }

    /// <summary>
    /// 记录 NPC 当天发生的具体事件（Buffer）
    /// </summary>
    public static class NpcTodayActionBuffer
    {
        private static readonly object _lock = new();

        // 字典存储：Key = NPC名字, Value = 动作列表
        private static readonly Dictionary<string, List<TodayAction>> _actions = new();

        public static void Add(string npcName, string description)
        {
            lock (_lock)
            {
                if (!_actions.ContainsKey(npcName))
                {
                    _actions[npcName] = new List<TodayAction>();
                }

                _actions[npcName].Add(new TodayAction
                {
                    Description = description,
                    Consumed = false
                });
            }
        }

        public static List<TodayAction> ConsumeAll(string npcName)
        {
            lock (_lock)
            {
                if (!_actions.ContainsKey(npcName))
                    return new List<TodayAction>();

                var fresh = _actions[npcName].Where(a => !a.Consumed).ToList();
                foreach (var a in fresh) a.Consumed = true;
                return fresh;
            }
        }

        public static List<string> GetAllActiveNpcNames()
        {
            lock (_lock)
            {
                return _actions.Keys.ToList();
            }
        }

        public static void ClearAtDayEnd()
        {
            lock (_lock)
            {
                _actions.Clear();
            }
        }
    }

    /// <summary>
    /// NPC Attitude 系统（事件驱动 + 时间衰减）
    /// </summary>
    public static class NpcAttitudeSystem
    {
        // ✅ 核心修复：使用字典存储每个 NPC 独立的 Attitude 值，而不是全局共享一个变量
        private static Dictionary<string, float> _attitudeValues = new();

        /// <summary>
        /// 获取特定 NPC 的当前离散姿态
        /// </summary>
        public static NpcAttitude GetCurrentAttitude(string npcName)
        {
            float val = GetRawValue(npcName);
            return MapToAttitude(val);
        }

        /// <summary>
        /// 获取原始浮点值 (仅内部或调试用)
        /// </summary>
        private static float GetRawValue(string npcName)
        {
            return _attitudeValues.ContainsKey(npcName) ? _attitudeValues[npcName] : 0f;
        }

        public static bool DebugMode { get; set; } = true;

        #region ===== Event Hooks (外部调用接口) =====

        /// <summary>
        /// 送礼反馈
        /// </summary>
        public static void OnGiftReaction(string npcName, GiftReaction reaction)
        {
            float oldValue = GetRawValue(npcName);
            float delta = 0f;

            switch (reaction)
            {
                case GiftReaction.Love: delta = +0.25f; break;
                case GiftReaction.Like: delta = +0.10f; break;
                case GiftReaction.Neutral: delta = +0.05f; break;
                case GiftReaction.Dislike: delta = -0.15f; break;
                case GiftReaction.Hate: delta = -0.30f; break;
            }

            ApplyDelta(npcName, delta);

            if (DebugMode)
            {
                LogAttitudeChange(npcName, $"送礼反馈({reaction})", oldValue, delta, GetRawValue(npcName));
            }
        }

        /// <summary>
        /// 翻垃圾桶被抓
        /// </summary>
        public static void OnTrashRummagedCaught(string npcName)
        {
            float oldValue = GetRawValue(npcName);
            float delta = -0.40f;

            ApplyDelta(npcName, delta);

            if (DebugMode)
            {
                LogAttitudeChange(npcName, "翻垃圾桶被抓", oldValue, delta, GetRawValue(npcName));
            }
        }

        /// <summary>
        /// 睡觉 → 新的一天开始（对所有记录在案的 NPC 进行情绪归零衰减）
        /// </summary>
        public static void OnDayStarted()
        {
            var knownNpcs = new List<string>(_attitudeValues.Keys);
            foreach (var name in knownNpcs)
            {
                DecayTowardNeutral(name);
            }
        }

        #endregion

        #region ===== Internal Logic (内部逻辑) =====

        private static void ApplyDelta(string npcName, float delta)
        {
            float oldValue = GetRawValue(npcName);
            float newValue = Clamp(oldValue + delta);
            _attitudeValues[npcName] = newValue;
        }

        private static void DecayTowardNeutral(string npcName)
        {
            float oldValue = GetRawValue(npcName);

            // 乘以衰减因子 (0.5 表示每天情绪减淡一半)
            float newValue = oldValue * 0.5f;

            // 归零阈值：非常接近 0 时直接重置为 0
            if (MathF.Abs(newValue) < 0.05f)
                newValue = 0f;

            _attitudeValues[npcName] = newValue;

            if (DebugMode && MathF.Abs(oldValue - newValue) > 0.001f)
            {
                LogDebug($"[{npcName}] 每日衰减: {oldValue:F3} -> {newValue:F3}");
            }
        }

        private static float Clamp(float value)
        {
            return MathF.Max(-1f, MathF.Min(1f, value));
        }

        private static NpcAttitude MapToAttitude(float value)
        {
            if (value < -0.6f) return NpcAttitude.Angry;
            if (value < -0.2f) return NpcAttitude.Irritated;
            if (value <= 0.2f) return NpcAttitude.Neutral;
            if (value <= 0.6f) return NpcAttitude.Pleasant;
            return NpcAttitude.Warm;
        }

        #endregion

        #region ===== Debug Methods =====

        private static void LogAttitudeChange(string npcName, string eventName, float oldValue, float delta, float newValue)
        {
            NpcAttitude newAlt = MapToAttitude(newValue);
            // 格式: [Attitude] [Damon] 送礼反馈: 0.000 + 0.250 -> 0.250 (Warm)
            string message = $"[Attitude] [{npcName}] {eventName}: " +
                             $"{oldValue:F3} + {delta:F3} → {newValue:F3} ({newAlt})";

            LogDebug(message);
        }

        private static void LogDebug(string message)
        {
            // 如果是 SMAPI 环境，建议替换为 Monitor.Log
            Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {message}");
        }

        // 调试用：手动设置某人的数值
        public static void DebugSetAttitude(string npcName, float value)
        {
            _attitudeValues[npcName] = Clamp(value);
        }

        #endregion
    }
}