import math
import os
import cv2
import hydra
import tonic
import torch
from tonic.transforms import Optional, ToFrame
from datasets.utils.pad_tensors import PadTensors
from datasets.utils.diskcache import DiskCachedDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np
from pandas import DataFrame, read_csv

sensor_size = (320, 320, 2)

file_list = ["120cmpm.csv", "30cmpm.csv", "60cmpm.csv"]

csv_path = "../../datasets/welding-data/recording1.csv"
df = read_csv(csv_path, header=None, names=["x", "y", "p", "t"])
event_data = df.to_numpy()
print(f"Event data shape: {event_data.shape}")

# transforming the events in the numpy array into tonic compatible events, with the format (x, y, p, t) and the dtype for each entry "<i8"
transformed_events = np.zeros((event_data.shape[0],), dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")])
transformed_events["x"] = event_data[:, 0]
transformed_events["y"] = event_data[:, 1]
transformed_events["p"] = event_data[:, 2]
transformed_events["t"] = event_data[:, 3]
print(f"Transformed events shape: {transformed_events}")
event_data = transformed_events

# great now we're gonna use the to frame transformer to convert the raw data into frames and visualize them
_event_to_tensor = ToFrame(sensor_size=sensor_size, time_window=50000)
frames = _event_to_tensor(event_data)
print(f"Frames shape: {frames.shape}")

video_path = "welding_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (sensor_size[0], sensor_size[1]))

for i in range(frames.shape[0]):
    # Sum positive and negative event channels → shape (320, 320)
    frame = frames[i].sum(axis=0).astype(np.float32)

    frame = np.log1p(frame)  # log(1 + x) handles zeros safely
    frame_min, frame_max = frame.min(), frame.max()
    if frame_max > frame_min:
        frame = (frame - frame_min) / (frame_max - frame_min) * 255.0
    frame = frame.astype(np.uint8)

    # Convert single-channel to 3-channel BGR (required by VideoWriter)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    out.write(frame_bgr)

out.release()
print(f"Video saved to {video_path}")