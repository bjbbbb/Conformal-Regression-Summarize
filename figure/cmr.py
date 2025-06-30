import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
from datetime import datetime

# 导入简化版CNFCP实现
from torchcp.regression.predictors.cmr import run_cnfcp

# 直接导入数据集构建函数
from examples.common.dataset import build_reg_data
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


def run_experiments(data_names, ratio_train=0.7, test_ratio=0.2, alpha=0.1, seed=1):
    """运行CNFCP实验并生成可视化 - 不需要计算评估指标"""
    device = get_device()

    # 使用固定参数
    hidden_features = 16
    num_layers = 5
    epochs = 400
    batch_size = 100
    sigma = 0.01
    num_samples = 1000

    # 对每个数据集运行实验
    for data_name in data_names:
        print(f"\n========== 数据集: {data_name} ==========")

        # 加载数据
        X_train, X_calib, X_test, y_train, y_calib, y_test = build_reg_data(
            data_name, ratio_train, test_ratio, seed, normalize=False
        )
        print(f"数据维度: X_train: {X_train.shape}, y_train: {y_train.shape}")

        # 准备数据加载器
        cal_dataset = TensorDataset(torch.from_numpy(X_calib).float(), torch.from_numpy(y_calib).float())
        test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

        cal_data_loader = DataLoader(cal_dataset, batch_size=min(8, len(cal_dataset)), shuffle=False, pin_memory=True)
        test_data_loader = DataLoader(test_dataset, batch_size=min(8, len(test_dataset)), shuffle=False,
                                      pin_memory=True)

        start_time = time.time()

        # 运行CNFCP可视化（不计算评估指标）
        results = run_cnfcp(
            X_train=X_train,
            y_train=y_train,
            cal_data_loader=cal_data_loader,
            test_data_loader=test_data_loader,
            X_test=X_test,
            y_test=y_test,
            alpha=alpha,
            hidden_features=hidden_features,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            sigma=sigma,
            learning_rate=3e-3,
            num_samples=num_samples,
        )


if __name__ == '__main__':
    # 设置随机种子
    fix_randomness(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 定义要测试的数据集
    data_names = ['synthetic']

    # 固定参数
    ratio_train = 0.7
    test_ratio = 0.2
    alpha = 0.1

    # 运行实验
    print(f"开始运行CNFCP可视化实验，数据集: {', '.join(data_names)}")
    run_experiments(data_names, ratio_train, test_ratio, alpha)

    print("所有CNFCP可视化实验已完成!")