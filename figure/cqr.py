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

from examples.common.dataset import build_reg_data
from examples.common.utils import build_regression_model

from torchcp.regression.predictors import CQR
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


def save_test_predictions(predictor, test_data_loader, device, run_number=None):
    """
    测试集上保存预测结果：
      - x：一维数据的第一个特征
      - y：真实标签
      - within_interval：若预测区间覆盖了 y，则为 1，否则为 0
    """
    predictions_list = []
    for batch_x, batch_y in test_data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_predictions = predictor.predict(batch_x)
        if torch.is_tensor(batch_predictions):
            batch_predictions = batch_predictions.cpu().numpy()
        batch_x_np = batch_x.cpu().numpy()
        batch_y_np = batch_y.cpu().numpy()
        for x_i, y_i, pred in zip(batch_x_np, batch_y_np, batch_predictions):
            # 兼容预测结果为tuple、list或numpy数组（取前两个值作为下、上界）
            if isinstance(pred, (tuple, list)) and len(pred) >= 2:
                lower, upper = float(pred[0]), float(pred[1])
            elif isinstance(pred, np.ndarray) and pred.size >= 2:
                lower, upper = float(pred.flat[0]), float(pred.flat[1])
            else:
                lower, upper = -np.inf, np.inf

            indicator = 1 if (y_i >= lower and y_i <= upper) else 0
            predictions_list.append({
                'x': x_i[0],  # x为一维数据，取第一个特征
                'y': float(y_i),
                'within_interval': indicator
            })

    run_str = "" if run_number is None else f"_run{run_number}"
    file_name = f"cqr_test_predictions{run_str}.csv"
    pd.DataFrame(predictions_list).to_csv(file_name, index=False)
    print(f"测试集结果已保存到 {file_name}")


def run_cqr(X_train, train_data_loader, cal_data_loader, test_data_loader, X_test, y_test, alpha, device, epochs=100, run_number=None):
    print("运行CQR方法...")
    quantiles = [alpha / 2, 1 - alpha / 2]
    model = build_regression_model("NonLinearNet")(X_train.shape[1], 2, 64, 0.5).to(device)
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)

    model.eval()
    predictor = CQR(model)
    predictor.calibrate(cal_data_loader, alpha)
    save_test_predictions(predictor, test_data_loader, device, run_number=run_number)


def save_test_predictions(predictor, test_data_loader, device, run_number=None):
    """
    测试集上保存预测结果：
      - x：一维数据的第一个特征
      - y：真实标签
      - lower：预测区间下界
      - upper：预测区间上界
      - within_interval：若预测区间覆盖了 y，则为 1，否则为 0
    """
    predictions_list = []
    for batch_x, batch_y in test_data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_predictions = predictor.predict(batch_x)
        if torch.is_tensor(batch_predictions):
            batch_predictions = batch_predictions.cpu().numpy()
        batch_x_np = batch_x.cpu().numpy()
        batch_y_np = batch_y.cpu().numpy()
        for x_i, y_i, pred in zip(batch_x_np, batch_y_np, batch_predictions):
            # 兼容预测结果为tuple、list或numpy数组（取前两个值作为下、上界）
            if isinstance(pred, (tuple, list)) and len(pred) >= 2:
                lower, upper = float(pred[0]), float(pred[1])
            elif isinstance(pred, np.ndarray) and pred.size >= 2:
                lower, upper = float(pred.flat[0]), float(pred.flat[1])
            else:
                lower, upper = -np.inf, np.inf

            indicator = 1 if (y_i >= lower and y_i <= upper) else 0
            predictions_list.append({
                'x': x_i[0],  # x为一维数据，取第一个特征
                'y': float(y_i),
                'lower': lower,
                'upper': upper,
                'within_interval': indicator
            })

    run_str = "" if run_number is None else f"_run{run_number}"
    file_name = f"cqrm_test_predictions{run_str}.csv"
    pd.DataFrame(predictions_list).to_csv(file_name, index=False)
    print(f"测试集结果已保存到 {file_name}")

    # 生成可视化
    visualize_predictions(file_name, run_number)

# Correct approach using direct alpha handling with NaN or mask
def visualize_predictions(predictions_file, run_number=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(12, 8))

    # Load data
    predictions_df = pd.read_csv(predictions_file)

    # 根据真实值是否在预测区间内设置颜色
    in_interval = predictions_df['y'].between(predictions_df['lower'], predictions_df['upper'])
    colors = np.where(in_interval, 'red', 'black')

    # 绘制散点图（70%透明度）
    plt.scatter(predictions_df['x'], predictions_df['y'], facecolors='none',
                edgecolors=colors, s=50, alpha=0.7, label='Data Points')

    # Prepare grid
    predictions_df = predictions_df.sort_values('x')
    x_min, x_max = predictions_df['x'].min(), predictions_df['x'].max()
    y_min = min(predictions_df['lower'].min(), predictions_df['y'].min())
    y_max = max(predictions_df['upper'].max(), predictions_df['y'].max())
    grid_size = 1000
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    density_grid = np.zeros((grid_size, grid_size))

    x_width_factor = 0.01
    for _, row in predictions_df.iterrows():
        x_val = row['x']
        lower = row['lower']
        upper = row['upper']
        x_range = np.abs(x_grid - x_val) < (x_grid[1] - x_grid[0]) * x_width_factor * grid_size / (x_max - x_min)
        x_indices = np.where(x_range)[0]
        y_lower_idx = np.abs(y_grid - lower).argmin()
        y_upper_idx = np.abs(y_grid - upper).argmin()
        for x_idx in x_indices:
            density_grid[y_lower_idx:y_upper_idx + 1, x_idx] += 1

    # 创建colormap（直接使用原始计数，不标准化）
    colors = [
        (0.0, 0.6, 0.9),  # **深海蓝**（冷色系基准）
        (0.957, 0.569, 0.235),  # 亮珊瑚橙（原鲜明橙提亮）
        (0.2, 0.8, 0.2),
        (0.6, 0.349, 0.718),  # 深紫罗兰（原鲜明紫加深）
        (0.992, 0.859, 0.0),  # 荧光黄
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # 绘制密度图（不标准化，直接使用原始计数）
    mask = density_grid > 0
    plt.imshow(np.where(mask, density_grid, np.nan), extent=[x_min, x_max, y_min, y_max],
               origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=np.nanmax(density_grid))

    # 添加颜色条（显示原始计数）
    cbar = plt.colorbar()
    cbar.set_label('Overlap Count', fontsize=32)  # 修改标签为计数

    # 绘制浅灰色预测线
    for _, row in predictions_df.iterrows():
        plt.plot([row['x'], row['x']], [row['lower'], row['upper']], 'k-', alpha=0.1, linewidth=0.5)

    # 设置标签和标题
    plt.xlabel('X', fontsize=32)
    plt.ylabel('Y', fontsize=32)
    run_str = "" if run_number is None else f" (Run {run_number})"
    plt.title(f'CQR', fontsize=32)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # 保存为PDF格式
    run_suffix = "" if run_number is None else f"_run{run_number}"
    plt.savefig(f"cqr_synthetic.pdf", dpi=300, bbox_inches='tight', format='pdf')
    print(f"cqr{run_suffix}.pdf")
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

            if method_name == 'CQR':
                run_cqr(
                    X_train,
                    train_data_loader,
                    cal_data_loader,
                    test_data_loader,
                    X_test,
                    y_test,
                    alpha,
                    device,
                    epochs=100,
                    run_number=run + 1  # 每次运行保存结果文件名附上 run 序号
                )

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
    methods = ['CQR']

    num_runs =1  # 总运行次数
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