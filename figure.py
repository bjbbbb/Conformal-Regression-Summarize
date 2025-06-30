import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from examples.common.dataset import build_reg_data

# 配置参数
DATA_NAMES = [
    'bike', 'community', 'star', 'homes',
    'meps_19', 'meps_20', 'meps_21',
    'facebook_1', 'facebook_2',
    'bio', 'concrete', 'blog_data'
]
SAVE_DIR = './distribution_plots'
DPI = 300
PLOT_SIZE = (10, 6)


def create_distribution_plots():
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    for data_name in DATA_NAMES:
        try:
            # 加载数据集
            X_propertrain, X_calib, X_test, y_propertrain, y_calib, y_test = build_reg_data(
                data_name,
                ratio_train=0.8,  # 80% 数据作为训练集
                test_ratio=0.2,   # 20% 数据作为测试集
                seed=42,  # 固定随机种子
                normalize=True  # 假设我们进行标准化
            )

            # 合并训练集和校准集的目标变量
            y = np.concatenate((y_propertrain, y_calib))

            # 创建画布
            plt.figure(figsize=PLOT_SIZE)

            # 绘制分布图
            sns.histplot(y, kde=True, bins=30,
                         color='#1f77b4',  # 更改颜色为蓝色
                         edgecolor='black',
                         alpha=0.7, line_kws={'lw': 2})

            # 添加标注
            plt.xlabel('Target Value (y)', fontsize=24, labelpad=12)  # 放大字体
            plt.ylabel('Density', fontsize=24, labelpad=12)
            plt.title(f'Distribution of Target Values: {data_name.capitalize()}',
                      fontsize=24, pad=18)

            # 修改坐标轴刻度字体大小
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=16)

            # 优化布局并保存
            plt.tight_layout()
            plt.savefig(
                os.path.join(SAVE_DIR, f'{data_name}_distribution.pdf'),
                dpi=DPI,
                bbox_inches='tight'
            )
            plt.close()

            print(f"✅ Successfully created plot for {data_name}")

        except Exception as e:
            print(f"❌ Error processing {data_name}: {str(e)}")
            continue


if __name__ == '__main__':
    create_distribution_plots()
    print(f"\nAll plots saved to: {os.path.abspath(SAVE_DIR)}")