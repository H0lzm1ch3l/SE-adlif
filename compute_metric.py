import hydra
from omegaconf import DictConfig
def load_wave(wave_file):
    import torchaudio


    waveform, sample_rate = torchaudio.load(wave_file)
    return waveform

def load_chunk_map(filepath):
    """Loads the chunk map from a pickle file if it exists."""
    from pathlib import Path
    import pickle
    if Path(filepath).is_file():
        with open(filepath, "rb") as f:
            chunk_map = pickle.load(f)
        print(f"Chunk map loaded from {filepath}")
        return chunk_map
    return None
def compute_metric(path_to_model_output:str, wave_files_path:str):
    import torchaudio
    from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
    from pathlib import Path
    import torch
    import numpy as np
    m_si_snr = ScaleInvariantSignalNoiseRatio()
    path_to_model_output = Path(path_to_model_output)
    wave_files_path = sorted(list(Path(wave_files_path).rglob("*.wav")))
    
    numpy_files = sorted(list(path_to_model_output.rglob("*.npy")),
                         key=lambda path: int(path.stem))
    for i, numpy_file in enumerate(numpy_files):
        pred_wave = np.load(numpy_file)
        pred_wave = torch.tensor(pred_wave, dtype=torch.float32)
        target_wave = torchaudio.load(wave_files_path[i])[0].unsqueeze(0)
        m_si_snr(pred_wave.permute((0,2,1)), target_wave)
        print(m_si_snr.compute())

@hydra.main(config_path="config", config_name="comp_metrics", version_base=None)
def main(cfg: DictConfig):
    compute_metric(cfg.path_to_model_output, cfg.wave_files_path)

if __name__ == "__main__":
    main()
