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

data_path = "/raid/home/michael.siegl/datasets/welding-data"
file_list = ["120cmpm.csv", "60cmpm.csv", "30cmpm.csv"]
roi = [[1.0e7, 2.0e7], [0.65e7, 2.85e7], [1.27e8, 1.6e8]]

# only save data within temporal roi to remove the long stretch of nothing happening in the videos, which is not useful for training and also takes up a lot of memory

for i, file in enumerate(file_list):
    csv_path = os.path.join(data_path, "csv", file)
    df = read_csv(csv_path, header=None, names=["x", "y", "p", "t"])
    event_data = df.to_numpy()
    print(f"Event data shape for {file}: {event_data.shape}")
    
    roi_event_data = event_data[np.logical_and(event_data[:, 3] > roi[i][0], event_data[:, 3] < roi[i][1])]
    print(f"Event data shape for {file} after applying temporal ROI: {roi_event_data.shape}")
    
    transformed_events = np.zeros((roi_event_data.shape[0],), dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")])
    transformed_events["x"] = roi_event_data[:, 0]
    transformed_events["y"] = roi_event_data[:, 1]
    transformed_events["p"] = roi_event_data[:, 2]
    transformed_events["t"] = roi_event_data[:, 3]
    print(f"Transformed events shape for {file}: {transformed_events.shape}")   
    
    # save all events as .npy files in npy folder in a folder for their class with is the same name as the csv file but without the .csv extension
    npy_folder = os.path.join(data_path, "npy", file.split(".")[0])
    os.makedirs(npy_folder, exist_ok=True)
    npy_path = os.path.join(npy_folder, file.split(".")[0] + ".npy")
    np.save(npy_path, transformed_events)
    print(f"Saved transformed events for {file} to {npy_path}")
    
    # also create a video but in the codes folder "." so here, nowhere in the datafolder!
    # save video for every file in the file list
    _event_to_tensor = ToFrame(sensor_size=sensor_size, time_window=50000)
    frames = _event_to_tensor(transformed_events)
    print(f"Frames shape for {file}: {frames.shape}")   
    video_path = file.split(".")[0] + ".avi"
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

# csv_path = "../../datasets/welding-data/recording1.csv"
# df = read_csv(csv_path, header=None, names=["x", "y", "p", "t"])
# event_data = df.to_numpy()
# print(f"Event data shape: {event_data.shape}")

# # transforming the events in the numpy array into tonic compatible events, with the format (x, y, p, t) and the dtype for each entry "<i8"
# transformed_events = np.zeros((event_data.shape[0],), dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")])
# transformed_events["x"] = event_data[:, 0]
# transformed_events["y"] = event_data[:, 1]
# transformed_events["p"] = event_data[:, 2]
# transformed_events["t"] = event_data[:, 3]
# print(f"Transformed events shape: {transformed_events}")
# event_data = transformed_events

# # great now we're gonna use the to frame transformer to convert the raw data into frames and visualize them
# _event_to_tensor = ToFrame(sensor_size=sensor_size, time_window=50000)
# frames = _event_to_tensor(event_data)
# print(f"Frames shape: {frames.shape}")

# video_path = "welding_video.avi"
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(video_path, fourcc, 20.0, (sensor_size[0], sensor_size[1]))

# for i in range(frames.shape[0]):
#     # Sum positive and negative event channels → shape (320, 320)
#     frame = frames[i].sum(axis=0).astype(np.float32)

#     frame = np.log1p(frame)  # log(1 + x) handles zeros safely
#     frame_min, frame_max = frame.min(), frame.max()
#     if frame_max > frame_min:
#         frame = (frame - frame_min) / (frame_max - frame_min) * 255.0
#     frame = frame.astype(np.uint8)

#     # Convert single-channel to 3-channel BGR (required by VideoWriter)
#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#     out.write(frame_bgr)

# out.release()
# print(f"Video saved to {video_path}")