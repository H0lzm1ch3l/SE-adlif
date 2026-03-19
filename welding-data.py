import os
import cv2
import numpy as np
from pandas import read_csv
from tonic.transforms import ToFrame

sensor_size = (320, 320, 2)

data_path = "/raid/home/michael.siegl/datasets/welding-data"
file_list = ["120cmpm.csv", "60cmpm.csv", "30cmpm.csv"]
roi = [[1.0e7, 2.0e7], [0.65e7, 2.85e7], [1.27e8, 1.6e8]]

TIME_WINDOW_US = 50_000  # 50 ms in microseconds

for i, file in enumerate(file_list):
    class_name = file.split(".")[0]  # e.g. "120cmpm"
    print(f"\n{'='*50}")
    print(f"Processing {file}...")

    # --- Load CSV ---
    csv_path = os.path.join(data_path, "csv", file)
    df = read_csv(csv_path, header=None, names=["x", "y", "p", "t"])
    event_data = df.to_numpy()
    print(f"  Raw event count: {event_data.shape[0]:,}")

    # --- Temporal ROI ---
    mask = np.logical_and(event_data[:, 3] > roi[i][0], event_data[:, 3] < roi[i][1])
    roi_event_data = event_data[mask]
    print(f"  Event count after ROI: {roi_event_data.shape[0]:,}")

    # --- Time-based windowing ---
    t = roi_event_data[:, 3]
    t_start = t[0]
    t_end   = t[-1]
    window_starts = np.arange(t_start, t_end, TIME_WINDOW_US)
    n_windows = len(window_starts)
    print(f"  Splitting into {n_windows} windows of {TIME_WINDOW_US / 1000:.1f} ms each")

    # --- Output folder ---
    npy_folder = os.path.join(data_path, "npy", class_name)
    os.makedirs(npy_folder, exist_ok=True)

    saved = 0
    skipped = 0
    for w, ws in enumerate(window_starts):
        we = ws + TIME_WINDOW_US

        idx_start = np.searchsorted(t, ws, side="left")
        idx_end   = np.searchsorted(t, we, side="right")

        window_raw = roi_event_data[idx_start:idx_end]

        if window_raw.shape[0] == 0:
            skipped += 1
            continue

        window = np.zeros(
            (window_raw.shape[0],),
            dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")]
        )
        window["x"] = window_raw[:, 0]
        window["y"] = window_raw[:, 1]
        window["p"] = window_raw[:, 2]
        window["t"] = window_raw[:, 3]

        npy_path = os.path.join(npy_folder, f"{class_name}_w{w:05d}.npy")
        np.save(npy_path, window)
        saved += 1

    print(f"  Saved {saved} windows, skipped {skipped} empty windows → {npy_folder}")

#     # --- Video preview (full ROI sequence, written to current working directory) ---
#     print(f"  Generating video preview...")

#     full_structured = np.zeros(
#         (roi_event_data.shape[0],),
#         dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")]
#     )
#     full_structured["x"] = roi_event_data[:, 0]
#     full_structured["y"] = roi_event_data[:, 1]
#     full_structured["p"] = roi_event_data[:, 2]
#     full_structured["t"] = roi_event_data[:, 3]

#     _event_to_tensor = ToFrame(sensor_size=sensor_size, time_window=TIME_WINDOW_US)
#     frames = _event_to_tensor(full_structured)
#     print(f"  Frames shape: {frames.shape}")

#     video_path = class_name + ".avi"
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(video_path, fourcc, 20.0, (sensor_size[0], sensor_size[1]))

#     for f_idx in range(frames.shape[0]):
#         frame = frames[f_idx].sum(axis=0).astype(np.float32)
#         frame = np.log1p(frame)
#         frame_min, frame_max = frame.min(), frame.max()
#         if frame_max > frame_min:
#             frame = (frame - frame_min) / (frame_max - frame_min) * 255.0
#         frame = frame.astype(np.uint8)
#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         out.write(frame_bgr)

#     out.release()
#     print(f"  Video saved to {video_path}")

# print(f"\nDone.")