import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# 更接近用户示例的风格
plt.style.use("seaborn-v0_8")
# 图注显示中文与负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 分段阶梯的节点（区间端点/代表值）
x_step = np.array([0,20, 50, 100, 150, 200, 550], dtype=float)
y_step = np.array([13,15, 17, 19, 21, 23, 23], dtype=float)

# 阶段中点作为控制点（不必严格经过端点），三次平滑样条
x_ctrl = np.array([15, 35, 75, 100, 200, 400,500], dtype=float)
y_ctrl = np.array([13, 15, 17, 18.5, 20.5, 23,24], dtype=float)
spline = UnivariateSpline(x_ctrl, y_ctrl, k=2, s=1)  # s 为平滑因子
x_smooth = np.linspace(x_step.min(), x_step.max(), 500)
y_smooth = spline(x_smooth)

fig, ax = plt.subplots(figsize=(9, 4.6))

# 阶梯状原函数
ax.step(x_step, y_step, where="post", label="分段函数", linewidth=2.2, color="#1f78b4")

# 样条平滑曲线
ax.plot(
    x_smooth,
    y_smooth,
    label="插值曲线",
    linewidth=2.2,
    color="#f28e2c",
    linestyle="--",
)

# 插值曲线 * 0.6
y_smooth_scaled = y_smooth * 0.6
ax.plot(
    x_smooth,
    y_smooth_scaled,
    label="插值曲线 × w",
    linewidth=2.2,
    color="#e15759",
    linestyle="-.",
)

# 节点散点，方便对照
ax.scatter(x_step, y_step, color="#444444", zorder=5, s=35)

ax.set_xlabel("进水浊度 (NTU)")
ax.set_ylabel("矾液药耗 (mg/L)")
ax.set_title("进水浊度与矾液药耗关系图")
ax.legend()
ax.grid(True, alpha=0.35)
ax.set_xlim(0, 550)
ax.set_ylim(7, 24.5)

plt.tight_layout()
plt.show()

# 计算在进水浊度65.2 NTU时的药耗
turbidity = 65.2
dose = spline(turbidity)
print(f"当进水浊度为 {turbidity} NTU 时，预测的矾液药耗为: {dose:.2f} mg/L")