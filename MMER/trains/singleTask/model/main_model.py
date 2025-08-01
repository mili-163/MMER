import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.markers import MarkerStyle
from numpy.random import multivariate_normal as mvn

np.random.seed(42)

# --------------------- 样本设置 ---------------------
n_samples = 500
colors = ['orange', 'teal']

# 新的分布生成（更自然）
def gen_natural_cluster(center, scale=1.0, angle_deg=0, elongation=1.0):
    angle_rad = np.deg2rad(angle_deg)
    rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    cov_matrix = np.diag([scale * elongation, scale])
    cov_rotated = rotation @ cov_matrix @ rotation.T
    return mvn(center, cov_rotated, size=n_samples)

# --------------------- 样本中心 ---------------------
local_centers = np.array([[0, 0], [5, 0]])              # 图(a)(b)
global_centers = np.array([[2, -6], [4, -2]])           # 图(c)(d) 斜角排列

# 锚点偏移
np.random.seed(10)
anchor_offset_a = np.random.uniform(-2.0, 2.0, size=(2, 2))
np.random.seed(30)
anchor_offset_c = np.random.uniform(-1.5, 2, size=(2, 2))
np.random.seed(20)
anchor_offset_b = np.random.uniform(-0.1, 0.1, size=(2, 2))
np.random.seed(40)
anchor_offset_d = np.random.uniform(-0.1, 0.1, size=(2, 2))

local_anchors_a = local_centers + anchor_offset_a
local_anchors_b = local_centers + anchor_offset_b
global_anchors_c = global_centers + anchor_offset_c
global_anchors_d = global_centers + anchor_offset_d

# --------------------- 工具函数 ---------------------
def add_solid_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle='->', linestyle='--',
                            linewidth=1.5, color='black', mutation_scale=8)
    ax.add_patch(arrow)

def add_gray_star(ax, point):
    ax.scatter(*point, marker=MarkerStyle('*'), s=30,
               facecolor='none', edgecolor='gray', linewidth=1.2)

# --------------------- 绘图准备 ---------------------
fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.5))
titles = [
    "(a) Single-modal - Existing\nMissing Rate=50%",
    "(b) Single-modal - AMP‑Distill\nMissing Rate=50%",
    "(c) Fusion - Existing\nMissing Rate=50%",
    "(d) Fusion - AMP‑Distill\nMissing Rate=50%"
]

for idx, ax in enumerate(axes.flatten()):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titles[idx], fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

# --------------------- (a) 更自然、更松散、旋转 ---------------------
ax = axes[0, 0]
for i, center in enumerate(local_centers):
    angle = 20 if i == 0 else -25
    scale = 1.2 if i == 0 else 1.0
    pts = gen_natural_cluster(center + np.random.randn(2) * 0.3, scale=scale, angle_deg=angle, elongation=1.6)
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, s=3, color=colors[i])
    mean_pt = pts.mean(axis=0)
    ax.scatter(*local_anchors_a[i], marker='*', s=100, color='red' if i == 0 else 'blue')
    add_gray_star(ax, mean_pt)
    add_solid_arrow(ax, mean_pt, local_anchors_a[i])

# --------------------- (b) 更集中，不同scale ---------------------
ax = axes[0, 1]
for i, center in enumerate(local_centers):
    jitter_center = center + np.array([0.3 * (-1)**i, 0.1 * i])
    pts = gen_natural_cluster(jitter_center, scale=0.25 + i * 0.05, angle_deg=i * 15)
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, s=3, color=colors[i])
    mean_pt = pts.mean(axis=0)
    ax.scatter(*(local_anchors_b[i] + np.random.randn(2) * 0.02), marker='*', s=100,
               color='red' if i == 0 else 'blue')
    add_gray_star(ax, mean_pt)
    add_solid_arrow(ax, mean_pt, local_anchors_b[i])

# --------------------- (c) 融合，自然形状，不同旋转 ---------------------
ax = axes[1, 0]
for i, center in enumerate(global_centers):
    angle = -15 if i == 0 else 45
    scale = 1.1
    pts = gen_natural_cluster(center + np.random.randn(2) * 0.2, scale=scale, angle_deg=angle, elongation=1.8 - 0.4 * i)
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, s=3, color=colors[i])
    mean_pt = pts.mean(axis=0)
    ax.scatter(*global_anchors_c[i], marker='*', s=100, color='red' if i == 0 else 'blue')
    add_gray_star(ax, mean_pt)
    add_solid_arrow(ax, mean_pt, global_anchors_c[i])

# --------------------- (d) 更集中，斜对角排列 ---------------------
ax = axes[1, 1]
for i, center in enumerate(global_centers):
    offset = np.array([0.4 * (-1)**i, 0.2 * i])
    pts = gen_natural_cluster(center + offset, scale=0.3 if i == 0 else 0.4, angle_deg=-30 + i * 40)
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, s=3, color=colors[i])
    mean_pt = pts.mean(axis=0)
    ax.scatter(*(global_anchors_d[i] + np.random.randn(2) * 0.02), marker='*', s=100,
               color='red' if i == 0 else 'blue')
    add_gray_star(ax, mean_pt)
    add_solid_arrow(ax, mean_pt, global_anchors_d[i])

# --------------------- 图例 + 保存 ---------------------
fig.suptitle('Figure X. Multimodal Sentiment Recognition at 50% Missing Rate', fontsize=12)
handles = [
    plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='orange', markersize=7, label='Positive Samples'),
    plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='teal', markersize=7, label='Negative Samples'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Positive Anchor'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=10, label='Negative Anchor'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9)
plt.tight_layout(rect=[0, 0.06, 1, 0.94])

# ✅ 保存为高清 PNG
plt.savefig("multimodal_sentiment_50_missing.png", dpi=300)
plt.show()
