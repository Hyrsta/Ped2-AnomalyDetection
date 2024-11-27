import os
import cv2 as cv
import torch
import numpy as np
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from tqdm import tqdm

def extract_flows(root_dir, output_dir, model):
    # Iterate over each sequence in the dataset with tqdm
    sequences = sorted(os.listdir(root_dir))
    for seq_idx, seq in enumerate(tqdm(sequences, desc="Sequences", unit="sequence"), start=1):
        seq_dir = os.path.join(root_dir, seq)
        if not os.path.isdir(seq_dir):
            continue

        # Create an output directory for this sequence
        seq_output_dir = os.path.join(output_dir, seq)
        os.makedirs(seq_output_dir, exist_ok=True)

        # Get all frame filenames in the sequence
        img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))])
        num_frames = len(img_files)
        num_pairs = num_frames - 1  # Number of frame pairs to process

        # Use tqdm to display progress for frame pairs
        pair_iterator = tqdm(range(num_pairs), desc=f"Processing sequence {seq}", unit="pair", leave=False)
        for i in pair_iterator:
            img1_path = os.path.join(seq_dir, img_files[i])
            img2_path = os.path.join(seq_dir, img_files[i + 1])

            # Read images using OpenCV
            img1 = cv.imread(img1_path)
            img2 = cv.imread(img2_path)

            # Check if images are loaded correctly
            if img1 is None or img2 is None:
                tqdm.write(f"Failed to read images {img1_path} or {img2_path}")
                continue

            # Initialize IOAdapter
            io_adapter = IOAdapter(model, img1.shape[:2])

            # Prepare inputs for the model using IOAdapter
            inputs = io_adapter.prepare_inputs([img1, img2])

            # Compute optical flow
            with torch.no_grad():
                predictions = model(inputs)

            # Extract the flow data
            flows = predictions['flows']

            # Convert flow to RGB image for visualization
            flow_rgb = flow_utils.flow_to_rgb(flows)
            flow_rgb = flow_rgb[0, 0].permute(1, 2, 0).cpu().numpy()
            flow_rgb_bgr = cv.cvtColor(flow_rgb, cv.COLOR_RGB2BGR)
            flow_rgb_bgr_uint8 = (flow_rgb_bgr * 255).astype(np.uint8)

            # Save the visualized flow image
            flow_img_filename = os.path.join(seq_output_dir, f'flow_{i:06d}.png')
            cv.imwrite(flow_img_filename, flow_rgb_bgr_uint8)

            # Save the raw flow data
            flow_np = flows[0, 0].permute(1, 2, 0).cpu().numpy()
            flow_data_filename = os.path.join(seq_output_dir, f'flow_{i:06d}.npy')
            np.save(flow_data_filename, flow_np)

            # Optional: Update tqdm description for current pair
            pair_iterator.set_postfix(Pair=f"{i + 1}/{num_pairs}")
    
        flow_img_filename = os.path.join(seq_output_dir, f'flow_{i+1:06d}.png')
        flow_data_filename = os.path.join(seq_output_dir, f'flow_{i+1:06d}.npy')
        cv.imwrite(flow_img_filename, flow_rgb_bgr_uint8)
        np.save(flow_data_filename, flow_np)

    print("\nOptical flow extraction completed.")

if __name__ == '__main__':
    # Define your dataset root directory
    root_dir = './ped2/training/frames'  # Replace with your dataset path

    # Define the output directory to save optical flow results
    model_name = 'flownet2'
    output_dir = './pltflow/train_photos/' + model_name  # Replace with your desired output path
    os.makedirs(output_dir, exist_ok=True)

    # Load the optical flow model
    model = ptlflow.get_model(model_name, pretrained_ckpt='things')

    # # Move the model to the appropriate device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    extract_flows(root_dir, output_dir, model)
