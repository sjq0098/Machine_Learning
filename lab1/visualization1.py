import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 数据
k_values = [5, 7, 9, 13]
sk_acc = [0.9052, 0.9096, 0.9115, 0.9033]
sk_nmi = [0.8293, 0.8290, 0.8336, 0.8224]
sk_cen = [0.4288, 0.4137, 0.3986, 0.4313]
my_acc = [0.9171, 0.9247, 0.9247, 0.9159]
my_nmi = [0.8405, 0.8521, 0.8503, 0.8358]
my_cen = [0.3534, 0.3220, 0.3170, 0.3653]

x = np.arange(len(k_values))
width = 0.35
colors = ["#4C72B0", "#DD8452"]

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

def add_labels(ax, rects):
    for r in rects:
        h = r.get_height()
        ax.annotate(f'{h:.3f}', (r.get_x()+r.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# ACC
r1 = axs[0].bar(x - width/2, sk_acc, width, color=colors[0], alpha=0.85)
r2 = axs[0].bar(x + width/2, my_acc, width, color=colors[1], alpha=0.85)
axs[0].set_title("ACC"); axs[0].set_ylabel("Score")
axs[0].set_xticks(x); axs[0].set_xticklabels(k_values)
add_labels(axs[0], r1); add_labels(axs[0], r2)

# NMI
r3 = axs[1].bar(x - width/2, sk_nmi, width, color=colors[0], alpha=0.85)
r4 = axs[1].bar(x + width/2, my_nmi, width, color=colors[1], alpha=0.85)
axs[1].set_title("NMI"); axs[1].set_xticks(x); axs[1].set_xticklabels(k_values)
add_labels(axs[1], r3); add_labels(axs[1], r4)

# CEN
r5 = axs[2].bar(x - width/2, sk_cen, width, color=colors[0], alpha=0.85)
r6 = axs[2].bar(x + width/2, my_cen, width, color=colors[1], alpha=0.85)
axs[2].set_title("CEN"); axs[2].set_xticks(x); axs[2].set_xticklabels(k_values)
add_labels(axs[2], r5); add_labels(axs[2], r6)

# —— 关键：在右侧空白区域放“全局图例” —— #
# 先给右侧留白
plt.subplots_adjust(right=0.84)
# 自定义图例句柄（颜色与柱子一致）
legend_handles = [Patch(facecolor=colors[0], label='sklearn'),
                  Patch(facecolor=colors[1], label='KNN')]
# 把图例放到图像右侧中央（不重叠子图）
fig.legend(handles=legend_handles, loc='center left',
           bbox_to_anchor=(0.86, 0.5), frameon=False, fontsize=11)

plt.suptitle("KNN Performance Comparison (LOOCV)", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.84, 0.95])  # 与上面的 right=0.84 对齐
plt.show()

