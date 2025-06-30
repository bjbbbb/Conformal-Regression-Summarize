import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from examples.common.dataset import build_reg_data

# 配置参数
DATA_NAMES = [
    'facebook_2'
]
SAVE_DIR = './conditional_distribution_plots'
DPI = 300
PLOT_SIZE = (12, 8)

# 改进的配色方案 - 使用更现代的色系
COLOR_PALETTE = "plasma"  # 也可以尝试 "plasma", "magma", "inferno" 或 "cividis"

# 全局字体大小设置
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 16
CBAR_LABEL_FONT_SIZE = 18
CBAR_TICK_FONT_SIZE = 14
TITLE_FONT_SIZE = 24


def create_conditional_distribution_plots():
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.facecolor'] = 'white'  # 设置背景为白色

    for data_name in DATA_NAMES:
        try:
            X_propertrain, X_calib, X_test, y_propertrain, y_calib, y_test = build_reg_data(
                data_name, ratio_train=0.8, test_ratio=0.2, seed=42, normalize=True
            )

            X = np.concatenate((X_propertrain, X_calib))
            y = np.concatenate((y_propertrain, y_calib))

            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            df_X = pd.DataFrame(X, columns=feature_names)
            df_y = pd.Series(y, name='Target')

            # 使用随机森林计算特征重要性
            model = RandomForestRegressor(random_state=42)
            model.fit(df_X, df_y)
            importances = model.feature_importances_

            # 只选择最重要的1个特征
            top_n = 1
            top_features = np.argsort(importances)[-top_n:][::-1]

            for i, feat_idx in enumerate(top_features, 1):
                plt.figure(figsize=PLOT_SIZE)
                x_feature = df_X.iloc[:, feat_idx]

                # 绘制填充图
                sns.kdeplot(
                    data=pd.DataFrame({'X': x_feature, 'Y': df_y}),
                    x='X', y='Y',
                    fill=True,
                    cmap=COLOR_PALETTE,
                    levels=10,
                    thresh=0.001,
                    bw_method=0.5,
                    common_norm=False,
                    cbar=True,
                    cbar_kws={
                        'label': 'Density',
                        'shrink': 1,
                        'aspect': 30
                    }
                )

                # 单独绘制黑色等高线
                sns.kdeplot(
                    data=pd.DataFrame({'X': x_feature, 'Y': df_y}),
                    x='X', y='Y',
                    fill=False,
                    color='black',
                    levels=10,
                    linewidths=0.5,
                    bw_method=0.5,
                    common_norm=False
                )

                # 添加散点图显示数据点分布
                sns.scatterplot(
                    x=x_feature, y=df_y,
                    color='white',
                    s=10,
                    alpha=0.3,
                    edgecolor='none'
                )

                # 设置坐标轴标签和标题
                plt.xlabel(f'Most Important Feature {feat_idx}',
                           fontsize=28, labelpad=12)
                plt.ylabel('Target Value', fontsize=28, labelpad=12)
                plt.title(f'{data_name.capitalize()} - p(Y|X)',
                          fontsize=29, pad=18)

                # 设置坐标轴刻度字体大小
                plt.tick_params(axis='both', which='major', labelsize=28)
                plt.tick_params(axis='both', which='minor', labelsize=28)

                # 调整颜色条
                ax = plt.gca()
                cbar = ax.collections[0].colorbar
                cbar.set_label('Density', fontsize=28)
                cbar.ax.tick_params(labelsize=24)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(SAVE_DIR, f'{data_name}_top_feature_conditional_distribution.pdf'),
                    dpi=DPI,
                    bbox_inches='tight'
                )
                plt.close()

            print(f"✅ Successfully created plots for {data_name}")

        except Exception as e:
            print(f"❌ Error processing {data_name}: {str(e)}")
            continue


if __name__ == '__main__':
    create_conditional_distribution_plots()
    print(f"\nAll plots saved to: {os.path.abspath(SAVE_DIR)}")
