import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from quantile_forest import RandomForestQuantileRegressor
from torch.utils.data import TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from examples.chr.black_boxes import QNet
from examples.chr.black_boxes import QNet, QRF
from examples.chr.methods import CHR
from examples.common.dataset import build_reg_data
from examples.common.utils import build_regression_model

from torchcp.regression.predictors import CQR, CTI
from torchcp.regression.loss import QuantileLoss
from torchcp.utils import fix_randomness


# 确保CUDA可用性
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        # 设置CUDA性能优化
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    return device


def train(model, device, epoch, train_data_loader, criterion, optimizer):
    model.train()
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader):
        tmp_x = tmp_x.to(device)
        tmp_y = tmp_y.to(device).unsqueeze(dim=1)

        optimizer.zero_grad()
        outputs = model(tmp_x)
        loss = criterion(outputs, tmp_y)
        loss.backward()
        optimizer.step()


def run_cti_nn(X_train, y_train, cal_data_loader, test_data_loader, X_test, y_test, alpha, device, epochs=2000,
               lr=0.0005, dropout=0.1):
    print("运行CTI(NN)方法...")
    quantiles = np.arange(0.01, 1.0, 0.01)
    model = QNet(quantiles, num_features=X_train.shape[1], no_crossing=True, batch_size=X_train.shape[0],
                 dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1, verbose=False)
    model.fit(X_train, y_train)

    predictor = CTI(model, quantiles, upper=y_train.max(), lower=y_train.min())
    predictor.calibrate(cal_data_loader, alpha)
    save_test_predictions(predictor, test_data_loader, device)


def save_test_predictions(predictor, test_data_loader, device, run_number=None):
    """
    测试集上保存预测结果：
      - x：一维数据的第一个特征
      - y：真实标签
      - intervals：所有有效区间的列表，每个区间包含下界和上界
    """
    predictions_list = []

    for batch_x, batch_y in test_data_loader:
        batch_x = batch_x.to(device)
        batch_x_np = batch_x.cpu().numpy()
        batch_y_np = batch_y.cpu().numpy()

        # 获取预测的分位数值
        quantile_predictions = predictor._model.predict(batch_x_np, quantiles=predictor.quantiles)

        # 添加上下界
        full_predictions = np.hstack([
            np.ones((quantile_predictions.shape[0], 1)) * predictor.lower,
            quantile_predictions,
            np.ones((quantile_predictions.shape[0], 1)) * predictor.upper
        ])

        # 计算区间长度
        interval_lengths = np.diff(full_predictions, axis=1)

        # 创建有效区间的掩码(长度小于等于q_hat的区间)
        valid_intervals_mask = interval_lengths <= predictor.q_hat

        for i, (x_i, y_i, intervals, valid_mask) in enumerate(
                zip(batch_x_np, batch_y_np, full_predictions, valid_intervals_mask)):
            # 提取所有有效区间
            valid_intervals = []
            for j, is_valid in enumerate(valid_mask):
                if is_valid:
                    lower = float(intervals[j])
                    upper = float(intervals[j + 1])
                    valid_intervals.append([lower, upper])

            # 检查y是否在任何有效区间内
            within_interval = 0
            for lower, upper in valid_intervals:
                if y_i >= lower and y_i <= upper:
                    within_interval = 1
                    break

            predictions_list.append({
                'x': x_i[0],  # x为一维数据，取第一个特征
                'y': float(y_i),
                'intervals': valid_intervals,
                'within_interval': within_interval
            })

    run_str = "" if run_number is None else f"_run{run_number}"
    file_name = f"cti_rf_test_predictions{run_str}.csv"

    # 将intervals列表转换为字符串以便保存到CSV
    df = pd.DataFrame(predictions_list)
    df['intervals'] = df['intervals'].apply(lambda x: str(x))
    df.to_csv(file_name, index=False)
    print(f"测试集结果已保存到 {file_name}")

    # 生成可视化
    visualize_predictions(file_name, run_number)


def visualize_predictions(predictions_file, run_number=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import ast  # 用于将字符串转换回列表
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(12, 8))

    # 加载数据
    predictions_df = pd.read_csv(predictions_file)

    # 将字符串形式的intervals转换回列表
    predictions_df['intervals'] = predictions_df['intervals'].apply(lambda x: ast.literal_eval(x))

    # 确定每个点是否在其预测区间内
    def is_in_interval(row):
        y_val = row['y']
        for lower, upper in row['intervals']:
            if lower <= y_val <= upper:
                return True
        return False

    predictions_df['in_interval'] = predictions_df.apply(is_in_interval, axis=1)

    # 获取数据范围
    predictions_df = predictions_df.sort_values('x')
    x_min, x_max = predictions_df['x'].min(), predictions_df['x'].max()

    # 计算y_min和y_max，考虑所有区间
    all_bounds = []
    for intervals in predictions_df['intervals']:
        for interval in intervals:
            all_bounds.extend(interval)
    if all_bounds:  # 确保存在有效边界
        y_min = min(min(all_bounds), predictions_df['y'].min())
        y_max = max(max(all_bounds), predictions_df['y'].max())
    else:
        y_min, y_max = predictions_df['y'].min(), predictions_df['y'].max()

    # 分析区间特性
    interval_lengths = []
    for intervals in predictions_df['intervals']:
        for lower, upper in intervals:
            length = upper - lower
            if length > 0:  # 只考虑长度大于0的区间
                interval_lengths.append(length)

    # 计算区间长度的统计信息
    if interval_lengths:
        min_length = min(interval_lengths)
        max_length = max(interval_lengths)
        mean_length = np.mean(interval_lengths)
        median_length = np.median(interval_lengths)
        print(f"区间统计: 最小长度={min_length:.6f}, 最大长度={max_length:.6f}")
        print(f"区间统计: 平均长度={mean_length:.6f}, 中位数长度={median_length:.6f}")
    else:
        min_length = 0.001  # 设置一个默认值避免除零错误

    print(f"区间数量: {len(interval_lengths)}")

    # 固定使用适中的网格分辨率，避免计算问题
    grid_size = 1000
    print(f"使用固定网格大小: {grid_size}")

    # 创建网格
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    density_grid = np.zeros((grid_size, grid_size))

    # 使用固定的x宽度因子
    x_width_factor = 0.01
    print(f"使用x宽度因子: {x_width_factor}")

    # 计算每个y网格点对应的实际y值
    y_step = (y_max - y_min) / (grid_size - 1)

    # 为每个点的每个有效区间绘制密度
    for _, row in predictions_df.iterrows():
        x_val = row['x']
        valid_intervals = row['intervals']

        # 计算该点在x网格上的索引
        x_indices = np.where(np.abs(x_grid - x_val) < (x_max - x_min) * x_width_factor)[0]

        # 对每个有效区间增加密度
        for lower, upper in valid_intervals:
            if lower >= upper:  # 跳过无效区间
                continue

            # 确保区间边界在网格内
            lower_bound = max(y_min, lower)
            upper_bound = min(y_max, upper)

            # 找到最接近的网格索引
            lower_idx = max(0, min(grid_size , int((lower_bound - y_min) / y_step)))
            upper_idx = max(0, min(grid_size , int((upper_bound - y_min) / y_step)))

            # # 确保索引有效性并添加一个小边距
            # lower_idx = max(0, lower_idx)  # 向下扩展1个网格点
            # upper_idx = min(grid_size, upper_idx+1)  # 向上扩展1个网格点

            # 更新密度网格
            for x_idx in x_indices:
                if lower_idx <= upper_idx:  # 确保有效区间
                    density_grid[lower_idx:upper_idx, x_idx] += 1

    # 绘制数据点，根据是否在区间内使用不同颜色
    in_interval_points = predictions_df[predictions_df['in_interval']]
    out_interval_points = predictions_df[~predictions_df['in_interval']]

    plt.scatter(in_interval_points['x'], in_interval_points['y'], facecolors='none',
                edgecolors='red', s=50, alpha=0.7)
    plt.scatter(out_interval_points['x'], out_interval_points['y'], facecolors='none',
                edgecolors='black', s=50, alpha=0.7)

    # 创建颜色映射
    colors = [
        (0.0, 0.6, 0.9),  # **深海蓝**（冷色系基准）
        (0.957, 0.569, 0.235),  # 亮珊瑚橙（原鲜明橙提亮）
        (0.2, 0.8, 0.2),
        (0.6, 0.349, 0.718),  # 深紫罗兰（原鲜明紫加深）
        (0.992, 0.859, 0.0),  # 荧光黄
    ]
    cmap = LinearSegmentedColormap.from_list("light_blue_to_deep_purple", colors)

    # 检查密度值的范围
    nonzero_density = density_grid[density_grid > 0]
    if len(nonzero_density) > 0:
        print(
            f"密度统计: 最小非零值={nonzero_density.min()}, 最大值={density_grid.max()}, 平均非零值={nonzero_density.mean()}")
        print(f"密度超过25%占比: {(density_grid > density_grid.max() * 0.25).sum() / nonzero_density.size * 100:.2f}%")
        print(f"密度超过50%占比: {(density_grid > density_grid.max() * 0.5).sum() / nonzero_density.size * 100:.2f}%")
        print(f"密度超过75%占比: {(density_grid > density_grid.max() * 0.75).sum() / nonzero_density.size * 100:.2f}%")

    # 掩码零值以实现透明效果
    mask = density_grid > 0
    im = plt.imshow(np.where(mask, density_grid, np.nan), extent=[x_min, x_max, y_min, y_max],
                    origin='lower', aspect='auto', cmap=cmap)

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Overlap Count', fontsize=32)

    # 为每个点的所有有效区间绘制细线
    for _, row in predictions_df.iterrows():
        for lower, upper in row['intervals']:
            if lower < upper:  # 只绘制有效区间
                plt.plot([row['x'], row['x']], [lower, upper], 'k-', alpha=0.1, linewidth=0.5)

    # 设置标签和标题
    plt.xlabel('X', fontsize=32)
    plt.ylabel('Y', fontsize=32)
    run_str = "" if run_number is None else f" (Run {run_number})"
    plt.title(f'CTI(NN)', fontsize=32)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='red',
               markerfacecolor='none', markersize=10),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
               markerfacecolor='none', markersize=10)
    ]

    # 保存图像为PDF格式
    run_suffix = "" if run_number is None else f"_run{run_number}"
    plt.savefig(f"cti(nn)_synthetic.pdf", dpi=300, bbox_inches='tight', format='pdf')
    print(f"可视化结果已保存到 cti_nn.pdf")
    plt.close()


def run_method_experiments(method_name, data_name, ratio_train, test_ratio, alpha, num_runs, device):
    """
    与原程序结构一样：循环 num_runs 次，每轮调用 build_reg_data 获得数据，
    然后利用 CQRM 训练、校准、预测并保存测试集的 [x, y, within_interval]。
    """
    print(f"正在运行 {method_name} 方法于 {data_name} 数据集 (训练集比例={ratio_train}, 显著性水平={alpha})")

    for run in range(num_runs):
        print(f"  运行 {run + 1}/{num_runs}")
        try:
            # 释放GPU内存以避免内存泄漏
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            seed = run + 1
            X_train, X_calib, X_test, y_train, y_calib, y_test = build_reg_data(
                data_name, ratio_train, test_ratio, seed, normalize=False
            )

            # 将数据转换为张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_calib_tensor = torch.tensor(X_calib, dtype=torch.float32)
            y_calib_tensor = torch.tensor(y_calib, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

            # 准备数据集和数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            cal_dataset = TensorDataset(X_calib_tensor, y_calib_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=100, shuffle=True, pin_memory=True
            )
            cal_data_loader = torch.utils.data.DataLoader(
                cal_dataset, batch_size=100, shuffle=False, pin_memory=True
            )
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=100, shuffle=False, pin_memory=True
            )

            if method_name == 'CTI(NN)':
                run_cti_nn(X_train, y_train, cal_data_loader, test_data_loader, X_test, y_test, alpha, device)

        except Exception as e:
            print(f"运行出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"{method_name} 在 {data_name} 上的测试结果保存完毕.")


if __name__ == '__main__':
    # 获取设备
    device = get_device()

    # 固定随机种子以增加可复现性
    fix_randomness(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data_names = ['synthetic']
    methods = ['CTI(NN)']

    num_runs = 1  # 总运行次数
    ratio_train_values = [0.7]
    test_ratio = 0.2
    alpha_values = [0.1]

    for data_name in data_names:
        for ratio_train in ratio_train_values:
            for alpha in alpha_values:
                print(f"\n{'=' * 50}")
                print(f"开始 {methods[0]} 在 {data_name} 上的实验: 训练集比例={ratio_train}, 显著性水平={alpha}")
                print(f"{'=' * 50}")
                run_method_experiments(methods[0], data_name, ratio_train, test_ratio, alpha, num_runs, device)

    print("所有实验已完成!")