import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from model import convAE
from utils_new import Ped2Dataset
from tqdm import tqdm
import cv2
from ptlflow.utils.flow_utils import flow_to_rgb


def test(model, val_loader, save_video=True, video_save_path='output_videos'):
    """
    Tests the model on the validation dataset and generates videos for original flow,
    reconstructed flow, and MSE heatmap.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        save_video (bool, optional): Whether to save the output videos. Defaults to True.
        video_save_path (str, optional): Directory to save the output videos. Defaults to 'output_videos'.

    Returns:
        int: Status code (0 for success).
    """
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: No GPU available, using CPU.")

    model.to(device)

    # Initialize video writers
    if save_video:
        os.makedirs(video_save_path, exist_ok=True)
        original_video_writer = None
        generated_video_writer = None
        heatmap_video_writer = None

    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Testing"):
            flows, batch_labels = data  # flows shape: (batch_size, 2, H, W)
            flows = flows.to(device)

            # Forward pass
            outputs = model(flows)  # outputs shape: (batch_size, 2, H, W)

            # Move data to CPU and convert to numpy
            flows_cpu = flows.cpu().numpy()          # Shape: (batch_size, 2, H, W)
            outputs_cpu = outputs.cpu().numpy()      # Shape: (batch_size, 2, H, W)

            # Convert flows to RGB images for visualization using ptlflow's flow_to_rgb
            # flow_to_rgb expects flow data in (H, W, 2) format
            original_rgb = [flow_to_rgb(flow.transpose(1, 2, 0)) for flow in flows_cpu]
            generated_rgb = [flow_to_rgb(flow.transpose(1, 2, 0)) for flow in outputs_cpu]

            # Initialize video writers based on frame size
            if save_video and original_video_writer is None:
                height, width, _ = original_rgb[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                original_video_writer = cv2.VideoWriter(
                    os.path.join(video_save_path, 'original_video.mp4'), fourcc, 24, (width, height)
                )
                generated_video_writer = cv2.VideoWriter(
                    os.path.join(video_save_path, 'generated_video.mp4'), fourcc, 24, (width, height)
                )
                heatmap_video_writer = cv2.VideoWriter(
                    os.path.join(video_save_path, 'heatmap_video.mp4'), fourcc, 24, (width, height)
                )

            # Compute MSE between original and reconstructed flows
            # Convert to torch tensors for MSE computation
            flows_tensor = torch.from_numpy(flows_cpu).to(device)          # (batch_size, 2, H, W)
            outputs_tensor = torch.from_numpy(outputs_cpu).to(device)      # (batch_size, 2, H, W)

            mse = torch.nn.functional.mse_loss(outputs_tensor, flows_tensor, reduction='none')  # (batch_size, 2, H, W)
            mse = mse.mean(dim=1)  # Mean over the flow channels -> (batch_size, H, W)

            # Move MSE to CPU and convert to numpy
            mse_cpu = mse.cpu().numpy()  # (batch_size, H, W)

            # Normalize MSE for visualization
            mse_normalized = []
            for m in mse_cpu:
                m_min = m.min()
                m_max = m.max()
                if m_max - m_min > 1e-8:
                    m_norm = (m - m_min) / (m_max - m_min)
                else:
                    m_norm = np.zeros_like(m)
                mse_normalized.append(m_norm)

            # Create heatmaps from normalized MSE
            heatmaps = [cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET) for m in mse_normalized]

            # Write frames to videos
            if save_video:
                for i in range(len(original_rgb)):
                    original_frame = original_rgb[i]      # RGB image
                    generated_frame = generated_rgb[i]    # RGB image
                    heatmap_frame = heatmaps[i]           # BGR image from applyColorMap

                    # Convert RGB to BGR for OpenCV
                    original_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
                    generated_bgr = cv2.cvtColor(generated_frame, cv2.COLOR_RGB2BGR)

                    # Overlay heatmap on original frame
                    heatmap_overlay = cv2.addWeighted(original_bgr, 0.7, heatmap_frame, 0.3, 0)

                    # Write frames to respective videos
                    original_video_writer.write(original_bgr)
                    generated_video_writer.write(generated_bgr)
                    heatmap_video_writer.write(heatmap_overlay)

    # Release video writers
    if save_video:
        if original_video_writer is not None:
            original_video_writer.release()
        if generated_video_writer is not None:
            generated_video_writer.release()
        if heatmap_video_writer is not None:
            heatmap_video_writer.release()

    print("Testing and video generation completed.")
    return 0


if __name__ == "__main__":
    model = convAE(n_channel=2)
    model.load_state_dict(torch.load('pltflow/models/best/best_model.pth'))
    test_dir = Ped2Dataset(root_dir="pltflow/test_photos/flownet2", pattern='test')
    test_loader = DataLoader(test_dir, batch_size=32, shuffle=False)
    test(model, test_loader, save_video=True, video_save_path='pltflow/output_videos')
