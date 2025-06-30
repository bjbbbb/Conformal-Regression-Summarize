import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib import rcParams


def plot_ranking_heatmap(ranking_df, SAVE_DIR, DPI=300):
    """
    绘制排名矩阵热图

    参数:
    ranking_df: 包含方法排名的DataFrame
    SAVE_DIR: 保存目录
    DPI: 图像分辨率
    """
    # 设置全局字体为Times New Roman，并放大字体大小
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 16  # 基础字体大小

    plt.figure(figsize=(16, 10))

    # 绘制热图
    sns.heatmap(ranking_df,
                annot=True,
                cmap='RdYlBu',
                fmt='.0f')  # 调整注释字体大小

    # 调整刻度标签
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)

    plt.tight_layout()

    # 保存图像
    plt.savefig(f'{SAVE_DIR}/ranking_heatmap.pdf',
                dpi=DPI,
                bbox_inches='tight')
    plt.close()

    print("✅ 排名矩阵热图已生成。")

# 定义数据集和方法列表
datasets = [
    'bike', 'bio', 'blog_data', 'community', 'concrete',
    'facebook_1', 'facebook_2', 'homes', 'meps_19',
    'meps_20', 'meps_21', 'star'
]

methods = [
    'CTI(RF)', 'CTI(NN)', 'CHR(RF)', 'CHR(NN)',
    'CQR', 'CQRM', 'CQRR', 'CQRFM', 'LCP', 'Split'
]

# 从表格数据中提取排名信息（基于Size指标）
# 表格数据格式: [dataset][method] = (value, std)
data = {
    'bike': {
        'CTI(RF)': 1.049, 'CTI(NN)': 0.722, 'CHR(RF)': 1.128, 'CHR(NN)': 0.757,
        'CQR': 1.564, 'CQRM': 1.289, 'CQRR': 1.534, 'CQRFM': 1.313,
        'LCP': 0.740, 'Split': 1.355
    },
    'bio': {
        'CTI(RF)': 1.297, 'CTI(NN)': 1.480, 'CHR(RF)': 1.456, 'CHR(NN)': 1.578,
        'CQR': 2.007, 'CQRM': 1.984, 'CQRR': 2.009, 'CQRFM': 1.980,
        'LCP': 1.780, 'Split': 1.982
    },
    'blog_data': {
        'CTI(RF)': 0.797, 'CTI(NN)': 1.006, 'CHR(RF)': 1.596, 'CHR(NN)': 1.771,
        'CQR': 3.340, 'CQRM': 1.943, 'CQRR': 3.303, 'CQRFM': 2.053,
        'LCP': 2.300, 'Split': 1.429
    },
    'community': {
        'CTI(RF)': 1.679, 'CTI(NN)': 1.325, 'CHR(RF)': 1.636, 'CHR(NN)': 1.574,
        'CQR': 1.758, 'CQRM': 1.661, 'CQRR': 1.764, 'CQRFM': 1.674,
        'LCP': 2.069, 'Split': 2.183
    },
    'concrete': {
        'CTI(RF)': 0.974, 'CTI(NN)': 0.465, 'CHR(RF)': 0.942, 'CHR(NN)': 0.493,
        'CQR': 0.700, 'CQRM': 0.628, 'CQRR': 0.699, 'CQRFM': 0.629,
        'LCP': 0.466, 'Split': 0.607
    },
    'facebook_1': {
        'CTI(RF)': 1.048, 'CTI(NN)': 0.790, 'CHR(RF)': 1.559, 'CHR(NN)': 1.389,
        'CQR': 2.693, 'CQRM': 2.148, 'CQRR': 2.696, 'CQRFM': 2.367,
        'LCP': 1.925, 'Split': 2.156
    },
    'facebook_2': {
        'CTI(RF)': 1.014, 'CTI(NN)': 0.775, 'CHR(RF)': 1.547, 'CHR(NN)': 1.413,
        'CQR': 2.778, 'CQRM': 1.910, 'CQRR': 2.747, 'CQRFM': 1.847,
        'LCP': 1.883, 'Split': 2.140
    },
    'homes': {
        'CTI(RF)': 0.638, 'CTI(NN)': 0.517, 'CHR(RF)': 0.684, 'CHR(NN)': 0.538,
        'CQR': 0.847, 'CQRM': 0.728, 'CQRR': 0.826, 'CQRFM': 0.734,
        'LCP': 0.545, 'Split': 0.829
    },
    'meps_19': {
        'CTI(RF)': 1.703, 'CTI(NN)': 1.801, 'CHR(RF)': 2.333, 'CHR(NN)': 2.547,
        'CQR': 2.898, 'CQRM': 2.599, 'CQRR': 2.908, 'CQRFM': 2.622,
        'LCP': 3.001, 'Split': 3.061
    },
    'meps_20': {
        'CTI(RF)': 1.796, 'CTI(NN)': 1.882, 'CHR(RF)': 2.357, 'CHR(NN)': 2.515,
        'CQR': 2.879, 'CQRM': 2.745, 'CQRR': 2.942, 'CQRFM': 2.718,
        'LCP': 3.231, 'Split': 3.052
    },
    'meps_21': {
        'CTI(RF)': 1.751, 'CTI(NN)': 1.844, 'CHR(RF)': 2.474, 'CHR(NN)': 2.667,
        'CQR': 2.927, 'CQRM': 2.714, 'CQRR': 2.934, 'CQRFM': 2.689,
        'LCP': 3.217, 'Split': 2.948
    },
    'star': {
        'CTI(RF)': 0.185, 'CTI(NN)': 0.193, 'CHR(RF)': 0.179, 'CHR(NN)': 0.207,
        'CQR': 0.181, 'CQRM': 0.179, 'CQRR': 0.181, 'CQRFM': 0.178,
        'LCP': 0.199, 'Split': 0.177
    }
}

# 创建排名矩阵
ranking_data = {}
for ds in datasets:
    # 获取当前数据集的所有方法值
    values = {method: data[ds][method] for method in methods}
    # 按值从小到大排序（值越小排名越高）
    sorted_methods = sorted(values.items(), key=lambda x: x[1])
    # 生成排名（1-based）
    rankings = {method: i+1 for i, (method, _) in enumerate(sorted_methods)}
    ranking_data[ds] = [rankings[method] for method in methods]

# 创建DataFrame
ranking_df = pd.DataFrame(ranking_data, index=methods).T

# 设置保存目录
SAVE_DIR = './combined_plots'
os.makedirs(SAVE_DIR, exist_ok=True)

# 运行函数
plot_ranking_heatmap(ranking_df, SAVE_DIR)