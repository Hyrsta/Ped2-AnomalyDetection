# Anomaly Detection on UCSD Ped2 Dataset Using FlowNet2.0 and AutoEncoder

## Project Overview
This project focuses on detecting anomalies in the UCSD Ped2 dataset by leveraging FlowNet2.0 for optical flow extraction and an AutoEncoder for modeling normal movement patterns. By learning typical motion behaviors, the system can effectively identify deviations that indicate anomalous activities.

## Framework
The workflow of this project consists of the following steps:
1. Optical Flow Extraction:
Utilize FlowNet2.0 to compute the optical flow between each pair of consecutive frames in the UCSD Ped2 dataset. This step captures the motion information essential for understanding normal and abnormal activities.
2. AutoEncoder Training:
Feed the extracted optical flow data into an AutoEncoder model. The AutoEncoder learns to reconstruct normal movement patterns, effectively modeling the typical behavior observed in the dataset.
3. Anomaly Detection:
During inference, the AutoEncoder attempts to reconstruct the optical flow of new frames. Anomalies are detected based on the reconstruction error—the assumption being that abnormal movements will result in higher reconstruction errors compared to normal movements.
