import hydra
from omegaconf import DictConfig

def compute_metric(path_to_model_output:str, wave_files_path:str, save_to: str):
    import torchaudio
    from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
    from pathlib import Path
    import torch
    import csv
    import numpy as np
    m_si_snr = ScaleInvariantSignalNoiseRatio()
    output_file = Path(save_to)
    output_file.mkdir(exist_ok=True, parents=True)
    path_to_model_output = Path(path_to_model_output)
    wave_files_path = sorted(list(Path(wave_files_path).rglob("*.wav")))
    
    numpy_files = sorted(list(path_to_model_output.rglob("*.npy")),
                         key=lambda path: int(path.stem))
    # Open file in write mode to start fresh
    with open(output_file /"si_snr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Mean"])  # Write header
    for i, numpy_file in enumerate(numpy_files):
        pred_wave = np.load(numpy_file)
        pred_wave = torch.tensor(pred_wave, dtype=torch.float32)
        target_wave = torchaudio.load(wave_files_path[i])[0].unsqueeze(0)
        m_si_snr(pred_wave.permute((0,2,1)), target_wave)
        r = m_si_snr.compute().item()
        print(m_si_snr.total)
        
        with open(output_file /"si_snr.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, r])

@hydra.main(config_path="config", config_name="comp_si_snr", version_base=None)
def main(cfg: DictConfig):
    compute_metric(cfg.path_to_model_output, cfg.wave_files_path, cfg.save_to)

if __name__ == "__main__":
    main()
