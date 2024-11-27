from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from PIL import Image
import cv2

class Ped2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, pattern='train'):
        """
        Initializes the Ped2Dataset with optical flow .npy files and corresponding labels.

        :param root_dir: Root directory containing subdirectories of optical flow .npy files.
        :param transform: Optional transformations to apply to the optical flow data.
        :param pattern: Specifies the dataset split ('train' or 'test').
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pattern = pattern
        self.all_files = []
        self.labels = []

        # Traverse the root directory and collect all .npy file paths
        for subdir, _, files in os.walk(root_dir):
            for file in sorted(files):
                if file.endswith('.npy'):
                    file_path = os.path.join(subdir, file).replace("\\", "/")
                    self.all_files.append(file_path)

        # Assign labels based on the pattern
        if pattern == 'train':
            # Assuming 'train' data has labels as 0 (e.g., normal frames)
            self.labels = [0] * len(self.all_files)
        elif pattern == 'test':
            # Load labels from a .npy file
            # Ensure that 'frame_labels_ped2.npy' has the same length as 'all_files'
            labels_path = 'pltflow/frame_labels_ped2.npy'  # Update this path as needed
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found at {labels_path}")
            
            loaded_labels = np.load(labels_path)
            self.labels = loaded_labels[0].tolist()
            
            if len(self.labels) != len(self.all_files):
                raise ValueError("Number of labels does not match number of flow files.")
        else:
            raise ValueError("Pattern must be either 'train' or 'test'.")

    def __len__(self):
        """
        Returns the total number of optical flow samples.
        """
        return len(self.all_files)

    def __getitem__(self, idx):
        """
        Retrieves the optical flow data and its corresponding label.

        :param idx: Index of the sample to retrieve.
        :return: A tuple of (flow_data, label).
                 - flow_data: Torch tensor of shape (2, H, W).
                 - label: Torch tensor containing the label.
        """
        flow_path = self.all_files[idx]
        label = self.labels[idx]

        # Load the optical flow data from the .npy file
        flow_np = np.load(flow_path)  # Shape: (H, W, 2)

        # Convert the numpy array to a torch tensor and rearrange dimensions to (C, H, W)
        flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1).float()  # Shape: (2, H, W)

        # Apply transformations if any
        if self.transform:
            flow_tensor = self.transform(flow_tensor)

        # Convert label to a torch tensor
        label_tensor = torch.tensor(label).long()

        return flow_tensor, label_tensor


if __name__ == '__main__':
    # 定义数据集和数据加载器
    transform = Compose([Resize((256, 256))])
    train_dir = 'pltflow/train_photos/flownet2'
    test_dir = 'pltflow/test_photos/flownet2'
    batch_size = 16

    train_dataset = Ped2Dataset(root_dir=train_dir, transform=transform, pattern='train')
    val_dataset = Ped2Dataset(root_dir=test_dir, transform=transform, pattern='test')
    test = train_dataset[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)