import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5]  # x 轴数据
y = [10, 15, 13, 18, 16]  # y 轴数据

# 绘制散点图
plt.scatter(x, y, color='blue', marker='o', s=50, alpha=0.7)
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")
plt.title("Scatter Plot Example")
plt.show()