现在不要马上继续改 selector。下一步应该做 真实质量评估。

也就是：不只看 selected 数量，而是看选出来的 examples 到底好不好。

1. 跑 Useful@3 人工评估
你现在已经能保证：


dynamic 不再被误过滤
接下来要看：


dynamic examples 是否真的适合作为 few-shot 风格参考
建议抽样：


每个 NPC 10 条 case
重点 NPC：Damon / Abigail / Shane / Sebastian / Penny
总计 50 条 case
每个 case 看最终进入 prompt 的 3-5 条 examples，人工标：

分数	含义
0	错角色 / 剧透 / 不可用
1	安全但不相关
2	可用，有风格参考
3	很适合当前场景

然后计算：


Useful@3
Mean usefulness score
Case Success Rate
这一步能回答：


dynamic few-shot 现在不仅能召回，而且召回得有用吗？
2. 重新跑 Valid@3
上一轮修复前后 Valid@3 是：


ALL: 58.3% → 68.8%
可攻略 NPC: 28.6% → 46.4%
现在改了 max_distance 后要重新跑一次。

重点看：


Valid@3 是否继续提升
可攻略 NPC 是否还是偏低
Damon 是否保持 100%
安全 violation 是否仍为 0
如果可攻略 NPC 还是低，就再看 selector 或 metadata，而不是继续调阈值。

3. 做 max_distance 阈值网格实验
现在 0.75 可用，但未必最优。

建议跑：


max_distance=None
max_distance=0.85
max_distance=0.80
max_distance=0.75
max_distance=0.70
max_distance=0.65
对比：


dynamic / mixed / static / empty 分布
selected>=3
Valid@3
Useful@3
Duplicate Rate
P95 latency
目标不是越严越好，而是找到平衡：


既不要召回太烂的样本，也不要把动态结果过滤到 fallback。
4. 检查 Penny 为什么是 mixed
现在 Penny/friend 是：


mixed, 2
这不一定是 bug，但值得检查。

可能原因：


1. Penny 的动态结果本来少
2. max_distance=0.75 对 Penny 偏严
3. Penny 的 metadata 或 language 分布不足
4. query 与 Penny 台词语义不够接近
5. selector 去重后只剩 2 条
检查方法：


看 raw_count
看被 max_distance 过滤掉的候选 distance
看被 relationship gate 过滤掉多少
看被 duplicate 去掉多少
如果只是个别 NPC mixed 没问题。
如果很多 NPC 都 mixed，说明阈值或语料覆盖还要调。

推荐下一轮目标
下一阶段可以定为：


Few-shot Quality Evaluation
交付物：


1. few_shot_eval_cases.jsonl，至少 50-100 条
2. run_few_shot_eval.py
3. max_distance 网格实验报告
4. Useful@3 人工标注表
5. Valid@3 / Useful@3 / fallback rate / latency 对比表
最终要形成这种结论：


在 max_distance=0.75 下，dynamic few-shot 覆盖率从 0 提升到 X%，Valid@3 为 Y%，Useful@3 为 Z%，安全 violation 仍为 0，P95 延迟保持在 145ms 左右。
这就是真正可展示的量化进步。