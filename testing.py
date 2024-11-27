import numpy as np
import os

array = np.eye(5)

model_save_dir = 'pltflow/models'
best_auc = float('-inf')
best_path = os.path.join(model_save_dir, 'best')
last_path = os.path.join(model_save_dir, 'last')
print(best_path)
os.makedirs(best_path, exist_ok=True)  # 确保目录存在
os.makedirs(last_path, exist_ok=True)  # 确保目录存在
best_model_path = best_path + '/best_model.pth'
last_model_path = last_path + 'last_model.pth'
best_normalized_loss = best_path + '/best_normalized_losses.npy'
last_normalized_loss = best_path + '/last_normalized_losses.npy'