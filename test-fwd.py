import numpy as np
from tonic import transforms as T
from torchvision.datasets import DatasetFolder
import matplotlib.pyplot as plt

from datasets.fwd import FWDLDM


data_path = "./data"
spatial_factor = 0.15
time_factor = 1.0
pad_to_min_size = 10
time_window = 500
num_classes = 3
ignore_first_timesteps = 1

fwdldm = FWDLDM(data_path=data_path, 
                spatial_factor=spatial_factor, 
                time_factor=time_factor, 
                time_window=time_window, 
                pad_to_min_size=pad_to_min_size, 
                num_classes=num_classes, 
                ignore_first_timesteps=ignore_first_timesteps)



# sensor_size = (320, 320, 2)

# data_path = "/raid/home/michael.siegl/datasets/welding-data/dataset"

# dataset = DatasetFolder(
#     root=data_path,
#     loader=np.load,
#     extensions=".npy",
#     transform=T.ToFrame(sensor_size=sensor_size, time_window=5000),
# )   
# for i in range(10):
#     frames, target = dataset[i]
#     print(f"Sample {i}:")
#     print(f"  Frames shape: {frames.shape}")
#     print(f"  Target: {target}")
#     # print(f"  First 5 frames:\n{frames[:5]}")   

# # plot all 10 frames of first sample 

# frames, target = dataset[0]
# print(f"Frames shape: {frames.shape}")  
# # create subplot for each frame
# fig, axes = plt.subplots(1, frames.shape[0], figsize=(20, 5))
# for t in range(frames.shape[0]):
#     frame = frames[t].sum(axis=0).astype(np.float32)
#     frame = np.log1p(frame)
#     frame_min, frame_max = frame.min(), frame.max()
#     if frame_max > frame_min:
#         frame = (frame - frame_min) / (frame_max - frame_min) * 255.0
#     frame = frame.astype(np.uint8)
#     axes[t].imshow(frame, cmap="gray")
#     axes[t].set_title(f"Frame {t}")
#     axes[t].axis("off")
# # save plot
# plt.savefig("fwd_sample.png")

