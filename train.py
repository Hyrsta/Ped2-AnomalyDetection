# 导入必要的库
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from model import convAE
import argparse
from utils_new import Ped2Dataset
from test import test
from torch import optim  # 导入 optim 模块
import logging

def setup_logging(log_file):
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def train(model, train_dir, test_dir, model_save_dir, epochs=10, batch_size=32, learning_rate=0.0001, logger=None):
    model.train()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logger.warning("No GPU available, using CPU.")
    else:
        logger.info(f"Using device: {device}")

    model.to(device)

    # 定义数据集和数据加载器
    transform = Compose([Resize((256, 256))])
    train_dataset = Ped2Dataset(root_dir=train_dir, transform=transform, pattern='train')
    val_dataset = Ped2Dataset(root_dir=test_dir, transform=transform, pattern='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=2)

    best_auc = float('-inf')
    best_path = os.path.join(model_save_dir, 'best')
    last_path = os.path.join(model_save_dir, 'last')
    print("best_path_is", best_path)
    os.makedirs(best_path, exist_ok=True)  # 确保目录存在
    os.makedirs(last_path, exist_ok=True)  # 确保目录存在
    best_model_path = best_path + '/best_model.pth'
    last_model_path = last_path + '/last_model.pth'
    best_normalized_loss = best_path + '/best_normalized_losses.npy'
    last_normalized_loss = last_path + '/last_normalized_losses.npy'

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for batch_id, data in enumerate(train_loader):
            images, _ = data  # 忽略标签
            images = images.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = torch.nn.functional.mse_loss(outputs, images)  # 使用均方误差作为损失函数

            if batch_id % 5 == 0:
                logger.info(f"Epoch: {epoch}, Batch: {batch_id}, Loss: {loss.item()}")

            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

        normalized_losses, auc = test(model, val_loader)
        normalized_losses_array = np.array(normalized_losses)
        # 指定保存路径
        save_path = 'pltflow/models/'
        os.makedirs(save_path, exist_ok=True)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to {best_model_path} with AUC: {auc}")

            # 保存为 .npy 文件
            np.save(best_normalized_loss, normalized_losses_array)
        else:
            torch.save(model.state_dict(), last_model_path)
            logger.info(f"Now model saved to {last_model_path} with AUC: {auc}")

            # 保存为 .npy 文件
            np.save(last_normalized_loss, normalized_losses_array)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Train an Autoencoder model")

    # 添加命令行参数
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training dataset directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the validation dataset directory')
    parser.add_argument('--model_save_dir', type=str, required=True, help='Directory to save the trained models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--log_file', type=str, default='training.log', help='Path to the log file')

    # 解析命令行参数
    args = parser.parse_args()

    # 配置日志记录
    logger = setup_logging(args.log_file)

    # 创建模型并训练
    model = convAE(n_channel=2)
    train(model, args.train_dir, args.test_dir, args.model_save_dir, args.epochs, args.batch_size, args.learning_rate, logger)
