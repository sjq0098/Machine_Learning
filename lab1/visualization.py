import matplotlib.pyplot as plt
import numpy as np

# 数据
k_values = [5, 7, 9, 13]
acc = [0.9171, 0.9247, 0.9247, 0.9159]
nmi = [0.8405, 0.8521, 0.8503, 0.8358]
cen = [0.3534, 0.3220, 0.3170, 0.3653]

x = np.arange(len(k_values))  # 横坐标位置
width = 0.25  # 每个柱子的宽度

fig, ax = plt.subplots(figsize=(8,5))

# 配色：蓝 / 绿 / 红（简洁科研风格）
colors = ["#4C72B0", "#55A868", "#C44E52"]

rects1 = ax.bar(x - width, acc, width, label='ACC', color=colors[0])
rects2 = ax.bar(x, nmi, width, label='NMI', color=colors[1])
rects3 = ax.bar(x + width, cen, width, label='CEN', color=colors[2])

# 标签
ax.set_xlabel("k")
ax.set_ylabel("Score")
ax.set_title("KNN Performance (LOOCV)")
ax.set_xticks(x)
ax.set_xticklabels(k_values)

# 图例：科研风格 → 右上角，无边框
ax.legend(loc='upper right', frameon=False)

# 在柱子上标数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()


