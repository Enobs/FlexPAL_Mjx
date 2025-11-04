import numpy as np
import matplotlib.pyplot as plt

import numpy as np

path = "/home/yinan/Documents/FlexPAL_Mjx/flexpal/flexpal/lpos_samples_abs_actions_mp.npz"   # 改成你的文件
D = np.load(path)
A = D["A"]          # (N, nu) 每步的动作（同一动作重复 T 次）
L = D["L"]          # (N, nu) 实际腱长
X = D["X"]          # (N, 3)
T = int(D["T"])
warmup = int(D["warmup"])
print("A shape:", A.shape, "nu:", A.shape[1], "T:", T, "warmup:", warmup)

# 1) 动作唯一组合与次数分布（理想情况下：每个唯一动作组合出现 exactly T-warmup 次）
uniqA, counts = np.unique(A, axis=0, return_counts=True)
print("唯一动作组合数:", len(uniqA))
print("counts 的最小/最大/平均:", counts.min(), counts.max(), counts.mean())
bad = np.where(counts != (T - warmup))[0]
print("次数不等于 T-warmup 的动作组合数:", len(bad))

# 如需看看哪几个动作不等于 T-warmup（通常为0）
if len(bad) > 0:
    print("前几个异常动作及出现次数：")
    for i in bad[:10]:
        print(uniqA[i], counts[i])

# 2) 各维度离散取值比例（检查 -1/0/1 是否大致 1/3）
print("\n各维度取值比例（全数据层面）：")
vals = np.unique(A)
for d in range(A.shape[1]):
    vs, cs = np.unique(A[:, d], return_counts=True)
    total = cs.sum()
    stat = {float(v): float(c/total) for v, c in zip(vs, cs)}
    # 为了可读性按常见的 -1,0,1 顺序打印
    out = [stat.get(-1.0, 0.0), stat.get(0.0, 0.0), stat.get(1.0, 0.0)]
    print(f"dim {d}:  P(-1)={out[0]:.3f}, P(0)={out[1]:.3f}, P(1)={out[2]:.3f}")

# 3) 每个动作组合内部（沿时间维）是否确实都是同一个动作（稳妥检查）
#    原理：按动作去重后，再按“块大小”是否一致来确认
ok_const = (counts == (T - warmup)).all()
print("\n同一唯一动作组合的样本数是否全为 T-warmup：", ok_const)

# 4) 抽查部分动作组合是否恰好对应 T-warmup 条记录（随机抽 k 个）
k = 5
rng = np.random.default_rng(0)
idx = rng.choice(len(uniqA), size=min(k, len(uniqA)), replace=False)
print("\n随机抽查几个动作组合的出现次数：")
for i in idx:
    m = (A == uniqA[i]).all(axis=1).sum()
    print("action=", uniqA[i], "count=", m)

# 5) 若你只用了 levels={-1,0,1}，理论总组合数应为 3^nu
levels = np.unique(A)
theory_total = (len(levels)) ** A.shape[1]
print(f"\n理论组合数（|levels|^nu）={theory_total}，实际唯一组合数={len(uniqA)}")

# 6) 简单看下 L 与 A 的关系是否“方向正确”
#    把 A 从 [-1,1] 映射到 ctrlrange 的目标 L*，看看 L 的分布是否逐步接近该范围端点/中点
L_min = L.min(axis=0)  # 粗略估计下全局最小/最大（如需更准，读 env 的 L_min/L_max）
L_max = L.max(axis=0)
print("\n(L 的粗略范围) per-dim min/max:")
for d in range(L.shape[1]):
    print(f"dim {d}: [{L_min[d]:.3f}, {L_max[d]:.3f}]")

# 7) 如果你要更严格：检验每个唯一动作组合的样本数“都一样”（等于 T-warmup）
print("\n是否均匀采样（定义：每个唯一动作组合出现次数相同）:", "YES" if counts.std() == 0 else "NO")
