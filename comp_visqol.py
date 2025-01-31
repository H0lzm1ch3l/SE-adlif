import hydra
from omegaconf import DictConfig
def compute_visqol(wave_files_path: str, path_to_model_output: str, save_to: str):
    from pathlib import Path
    import os
    from tqdm import tqdm
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2
    import csv
    import torchaudio
    import numpy as np
    config = visqol_config_pb2.VisqolConfig()

    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    wave_files_path = sorted(list(Path(wave_files_path).rglob("*.wav")))
    
    numpy_files = sorted(list(Path(path_to_model_output).rglob("*.npy")),
                         key=lambda path: int(path.stem))

    
    output_file = Path(save_to)
    output_file.mkdir(exist_ok=True, parents=True)

    # Open file in write mode to start fresh
    with open(output_file /"visqol.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Mean"])  # Write header

    r_sum = 0
    n = 0
    for i in tqdm(range(1, len(numpy_files)+1)):
        target_wave = torchaudio.load(wave_files_path[i-1])[0][0].numpy().squeeze()
        rec_data = np.load(numpy_files[i-1]).squeeze()
        if not(7*16_000 <target_wave.shape[0] < 11*16_000):
            continue
        try:
            
            similarity_result = api.Measure(target_wave.astype(np.float64), rec_data.astype(np.float64))
            r = similarity_result.moslqo
            n += 1
        
        except Exception as e: 
            continue
        r_sum += r
        r_mean = r_sum/n
        with open(output_file /"visqol.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, r_mean])
@hydra.main(config_path="config", config_name="comp_visqol", version_base=None)
def main(cfg: DictConfig):
    compute_visqol(cfg.wave_files_path, cfg.path_to_model_output, cfg.save_to)

if __name__ == "__main__":
    main()