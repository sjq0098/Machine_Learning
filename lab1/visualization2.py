import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 支持中文显示（若系统无 SimHei，可换成 'Microsoft YaHei'）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
k_values = [5, 7, 9, 13]
acc_orig = [0.9171, 0.9247, 0.9247, 0.9159]
nmi_orig = [0.8405, 0.8521, 0.8503, 0.8358]
cen_orig = [0.3534, 0.3220, 0.3170, 0.3653]

acc_aug  = [0.9726, 0.9676, 0.9640, 0.9563]
nmi_aug  = [0.9395, 0.9284, 0.9214, 0.9050]
cen_aug  = [0.1136, 0.1385, 0.1523, 0.1804]

# 颜色
c_orig, c_aug = "#4C72B0", "#DD8452"

# 画布：右侧预留空间给图例
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
fig.subplots_adjust(right=0.82)  # 右侧留白

def plot_one(ax, y1, y2, title, ylim=None, note=None):
    ax.plot(k_values, y1, marker="o", linewidth=2, label="原始数据", color=c_orig)
    ax.plot(k_values, y2, marker="s", linewidth=2, label="增强数据", color=c_aug)
    ax.set_title(title)
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # 数值标注
    for x, y in zip(k_values, y1):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8)
    for x, y in zip(k_values, y2):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=8)
    if note:
        ax.set_ylabel(note)

# 各子图
plot_one(axs[0], acc_orig, acc_aug, "ACC", ylim=(0.9, 1.0), note="Score")
plot_one(axs[1], nmi_orig, nmi_aug, "NMI", ylim=(0.80, 1.0))
plot_one(axs[2], cen_orig, cen_aug, "CEN（越低越好）", ylim=(0.10, 0.40))

# 右侧竖排图例（完全不遮挡）
legend_handles = [Patch(facecolor=c_orig, label="原始数据"),
                  Patch(facecolor=c_aug,  label="增强数据")]
fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(0.86, 0.5),
           frameon=False, fontsize=12)

plt.suptitle("KNN 在原始与增强数据上的性能对比（LOOCV）", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.82, 0.94])  # 和上面的 right=0.82 对齐
# 如需保存：plt.savefig("knn_aug_compare.png", dpi=300, bbox_inches="tight")
plt.show()


