import os
import cv2


model_name = 'flownet2'
# Define the directory where optical flow images are saved
optical_flow_dir = 'pltflow/output_photos/' + model_name  # Replace with your optical flow images directory

# Define the directory to save the generated videos
output_video_dir = 'pltflow/output_videos/' + model_name  # Replace with your desired output videos directory
os.makedirs(output_video_dir, exist_ok=True)

# Get a list of all sequence directories
sequences = sorted([seq for seq in os.listdir(optical_flow_dir) if os.path.isdir(os.path.join(optical_flow_dir, seq))])

# Loop over each sequence
for seq in sequences:
    seq_flow_dir = os.path.join(optical_flow_dir, seq)
    seq_video_output = os.path.join(output_video_dir, f'{seq}_optical_flow.mp4')

    # Get all image filenames in the sequence directory
    image_files = [f for f in os.listdir(seq_flow_dir) if f.endswith('.png')]
    image_files.sort()  # Ensure the files are in the correct order

    if not image_files:
        print(f"No images found in {seq_flow_dir}. Skipping...")
        continue

    # Read the first image to get frame dimensions
    first_frame_path = os.path.join(seq_flow_dir, image_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Failed to read the first image in {seq_flow_dir}. Skipping...")
        continue
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    fps = 24  # Adjust FPS as per your original video frame rate
    out = cv2.VideoWriter(seq_video_output, fourcc, fps, (width, height))

    # Write frames to the video
    for filename in image_files:
        img_path = os.path.join(seq_flow_dir, filename)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read {img_path}. Skipping this frame.")
            continue
        out.write(frame)

    out.release()
    print(f"Video saved to {seq_video_output}")

print("All videos have been generated.")
