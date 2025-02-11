import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
from tqdm import tqdm

def load_data(path_to_source: str, path_to_prediction: str):
    path_to_source: Path = Path(path_to_source)
    path_to_prediction: Path = Path(path_to_prediction)
    if not path_to_source.exists():
        raise FileNotFoundError(f'path to ground-truth waves {path_to_source} do not exist.')
    if not path_to_prediction.exists():
        raise FileExistsError(f'path to predicted waves {path_to_prediction} do not exist.')
    
    if (path_to_source.suffix == ".wav" and path_to_prediction.suffix == ".wav"):
        source_files = [path_to_source,]
        prediction_files = [path_to_prediction,]
    elif path_to_source.is_dir() and path_to_prediction.is_dir():
        source_files = list(sorted(path_to_source.rglob("*.wav")))
        prediction_files = list(sorted(path_to_prediction.rglob("*.wav")))
    if len(source_files) == 0:
        raise FileNotFoundError(f"uncompressed path {path_to_source} contains no '.wav' files")
    if len(prediction_files) == 0:
        raise FileNotFoundError(f"Prediction path {path_to_prediction} contains no '.wav' files")
    return source_files, prediction_files

def compute_visqol(source_files: list[Path,], prediction_files: list[Path,], 
                   sampling_rate: int = 24000):
    try:
        
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
        from visqol.pb2 import similarity_result_pb2
    except ImportError:
        print("Visqol was not found, please follow instructions in README in order to compile visqol")
        return
    import torchaudio
    import numpy as np
    import torch
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    r_mean = 0
    r_sum = 0
    n = 0
    for i in (pbar := tqdm(range(len(prediction_files)))):
        target_wave = torchaudio.load(str(source_files[i]))[0].squeeze()
        pred_wave = torchaudio.load(str(prediction_files[i]))[0].squeeze()
        
        # Following visqol guidelines we only evaluate waveforms that are 7-11 seconds long
        if not(7*sampling_rate < target_wave.shape[0] < 11*sampling_rate):
            continue
        # rescale
        target_wave = 1.0/(torch.max(target_wave.abs()))*target_wave
        
        # resample
        # visqol speech mode expect 16khz we will resample the waves files on the fly. 
        target_wave = torchaudio.functional.resample(
                target_wave, orig_freq=sampling_rate, new_freq=16_000,
                resampling_method='sinc_interp_kaiser'
            )
        pred_wave = torchaudio.functional.resample(
                pred_wave, orig_freq=sampling_rate, new_freq=16_000,
                resampling_method='sinc_interp_kaiser'
            )
        target_wave = target_wave.cpu().numpy().astype(np.float64)
        pred_wave = pred_wave.cpu().numpy().astype(np.float64)
        similarity_result = api.Measure(target_wave, pred_wave)
        r = similarity_result.moslqo
        n += 1
        r_sum += r
        r_mean = r_sum/n
        pbar.set_description(f'Running visqol mean: {r_mean}')
    print(f'Visqol mean : {r_mean}')
        
def compute_si_srn(source_files: list[Path,], prediction_files: list[Path,]):
    import torchaudio
    from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
    import torch
    from tqdm import tqdm
    m_si_snr = ScaleInvariantSignalNoiseRatio()
    r_mean = 0
    for i in (pbar := tqdm(range(len(prediction_files)))):
        target_wave = torchaudio.load(str(source_files[i]))[0]
        # rescaling
        target_wave = 1.0/torch.max(target_wave.abs())* target_wave
        
        pred_wave = torchaudio.load(str(prediction_files[i]))[0]
        r = m_si_snr(pred_wave, target_wave).item()
        
        r_mean = m_si_snr.compute().item()
        pbar.set_description(f'Running si-snr mean: {r_mean}')
    print(f'si-snr mean: {r_mean}')
        
@hydra.main(config_path="config", config_name="evaluate_metrics", version_base=None)
def main(cfg: DictConfig):
    sources_files, prediction_files = load_data(cfg.source_wave_path, cfg.pred_wave_path)
    
    if cfg.metric == "si_snr":
        compute_si_srn(sources_files, prediction_files)
    elif cfg.metric == "visqol":
        compute_visqol(sources_files, prediction_files, 24000)
    else:
        raise ValueError(f'metric name {cfg.metric} not understood \n options are: si_snr or visqol')

if __name__ == "__main__":
    main()