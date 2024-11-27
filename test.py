# 导入必要的库
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model import convAE
from sklearn.metrics import roc_auc_score
from utils_new import Ped2Dataset
import math
from tqdm import tqdm
import os


def psnr(mse):
    if isinstance(mse, np.ndarray):
        mse = mse.tolist()  # 将 NumPy 数组转换为列表
    if isinstance(mse, list):
        return [10 * math.log10(1 / m) for m in mse]
    return 10 * math.log10(1 / mse)

def sliding_window_mse(outputs, images, window_size, stride):
    """
    使用滑动窗口计算每个小块的MSE，并返回每个样本的最大MSE。

    参数:
    outputs (torch.Tensor): 模型输出，形状为 [batch_size, channels, height, width]
    images (torch.Tensor): 输入图像，形状为 [batch_size, channels, height, width]
    window_size (tuple): 滑动窗口的大小 (height, width)
    stride (int): 滑动窗口的步长

    返回:
    max_mses (torch.Tensor): 每个样本的最大MSE，形状为 [batch_size]
    """
    batch_size, channels, height, width = outputs.shape
    window_height, window_width = window_size
    max_mses = []

    for i in range(batch_size):
        output = outputs[i]
        image = images[i]
        max_mse = 0.0

        for y in range(0, height - window_height + 1, stride):
            for x in range(0, width - window_width + 1, stride):
                output_patch = output[:, y:y + window_height, x:x + window_width]
                image_patch = image[:, y:y + window_height, x:x + window_width]
                mse = torch.nn.functional.mse_loss(output_patch, image_patch, reduction='mean')
                if mse > max_mse:
                    max_mse = mse

        max_mses.append(max_mse)

    return torch.stack(max_mses).cpu().numpy()

def test(model, val_loader):
    model.eval()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: No GPU available, using CPU.")
    # else:
    #     # print(f"Using device: {device}")

    model.to(device)

    # 定义数据集和数据加载器
    #transform = Compose([Resize((224, 224)), ToTensor()])
    # transform = ToTensor()
    # test_dataset = Ped2Dataset(root_dir=test_dir, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 批量大小设置为32，以便批量处理图片
    test_loader = val_loader
    # 存储每张图片的损失值
    losses = []
    labels = []
    psnr_list = []
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, batch_labels = data  # 注意这里应该是 batch_labels 而不是 labels
            #输入图像数据：[32, 1, 240, 360]
            images = images.to(device)
            #print('输入图像:',images.shape)

            # 前向传播
            outputs = model(images)
            #输出图像形状：[32, 1, 240, 360]
            #print('输出图像：',outputs.shape)
            # 计算每个样本的损失
            batch_losses = sliding_window_mse(outputs, images, (24,36), 10)
            # batch_losses = torch.nn.functional.mse_loss(outputs, images, reduction='none').mean(
            #     dim=(1, 2, 3)).cpu().numpy()
            losses.extend(batch_losses)
            labels.extend(batch_labels.cpu().numpy().flatten())

    normalized_losses = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
    # normalized_losses = (np.max(psnr_list)-psnr_list) / (np.max(psnr_list) - np.min(psnr_list))

    # 检查标签列表
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise ValueError(f"Only one class present in y_true: {unique_labels}. ROC AUC score is not defined in that case.")
    normalized_losses_array = np.array(normalized_losses)
    labels_array = np.array(labels)
    # # 指定保存路径
    # save_path = '/home/dzs/MNAD/Dzc_bigwork/saved_models/10_24/npy'
    #
    # # 保存为 .npy 文件
    # np.save(save_path + 'normalized_losses.npy', normalized_losses_array)
    # np.save(save_path + 'labels.npy', labels_array)

    auc = roc_auc_score(labels, normalized_losses)
    print(f"Losses: {normalized_losses}")
    print(f"Labels: {labels}")
    print(f"AUC: {auc}")
    normalized_losses_array = np.array(normalized_losses)
    labels_array = np.array(labels)

    progress_path = 'pltflow/progress/'
    os.makedirs(progress_path, exist_ok=True)

    np.save(progress_path +'normalized_losses.npy', normalized_losses_array)
    np.save(progress_path + 'labels.npy', labels_array)
    return normalized_losses, auc

if __name__ == "__main__":
    # model_path = "/home/dzs/MNAD/Dzc_bigwork/saved_models/10_24/best_model/best_model.pth"
    model = convAE(n_channel=2)
    model.load_state_dict(torch.load('pltflow/models/best/best_model.pth'))
    # model.load_state_dict(torch.load(model_path))
    transform = Compose([Resize((256, 256))])
    test_dir = Ped2Dataset(root_dir=r"pltflow/test_photos", transform=transform,pattern='test')
    test_loader = DataLoader(test_dir, batch_size=32, shuffle=False)
    test(model, test_loader)