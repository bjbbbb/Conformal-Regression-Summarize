import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('./cqr_test_predictions_run1.csv')

# 提取x和y数据
x = df['x'].values
y = df['y'].values
within_interval = df['within_interval'].values


# 创建y值的网格
y_grid = np.linspace(0, 1, 2000)


plt.scatter(x[within_interval == 1], y[within_interval == 1], color='blue', label='Data Points (within interval)')
plt.scatter(x[within_interval == 0], y[within_interval == 0], color='gray', label='Data Points (outside interval)')

# 添加图例和标签
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# 显示图形
plt.show()