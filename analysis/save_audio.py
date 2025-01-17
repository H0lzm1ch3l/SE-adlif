from IPython.display import Audio
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from omegaconf import DictConfig, omegaconf
import torch
import torchaudio
from datasets.audio_compress import LibriTTS
from datasets.utils.diskcache import DiskCachedDataset
from models.pl_module_compress import MLPSNN
import os
import plotly.express as px
import plotly.graph_objects as go
import torchaudio


# %%
@torch.compiler.disable
def create_audio_example(cfg_path, ckpt_path, example_idx,return_spike_proba):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    cfg.unroll_factor = 1
    # cfg.compile=False
    # cfg.decoder.l_out.cell = 'li'
    # dataset = LibriTTS('/home/romain/datasets/LibriTTS/', sampling_freq=16_000, sample_length=-1, prediction_delay=cfg.dataset.prediction_delay)
    # dataset = LibriTTS(save_to='/calc/baronig/data_sets/LibriTTS/', cache_path='/calc/baronig/data_sets/LibriTTS/', sampling_freq=16_000, sample_length=-1)
    print("loading dataset")
    dataset = LibriTTS(save_to='/calc/baronig/data_sets/LibriTTS/', cache_path='/scratch/baronig/cache/librispeech', sampling_freq=16_000, sample_length=-1)
    print("loading checkpoint")
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    ckpt_state_dict = ckpt['state_dict']
    model = MLPSNN(cfg)
    model.load_state_dict(ckpt_state_dict)
    model.to("cuda:0")
    model.eval()
    source_waveform = []
    prediction_waveform = []
    spike_probs = []
    for idx in example_idx:
        print(f"Processing example {idx}")
        inputs, *rest = dataset[idx]
        source_waveform.append(inputs.cpu().numpy().squeeze())
        with torch.no_grad():
            if return_spike_proba:
                states, pred = model.forward_with_states(inputs.unsqueeze(0))
                # spike_prob = torch.mean(states[1][1]).item().cpu().numpy()
                spike_prob = torch.mean(states[1][1]).item()
                spike_probs.append(spike_prob)
            else:
                pred = model(inputs.to("cuda:0").unsqueeze(0))
            pred = pred[:, cfg.dataset.prediction_delay:]
            pred_wave = model.loss.generate_wave(pred)
        prediction_waveform.append(pred_wave.cpu().numpy().squeeze())
    return source_waveform, prediction_waveform, spike_probs

# %%
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return "model." + text[len(prefix) :]
    return text


def repair_checkpoint(path):
    ckpt = torch.load(path)
    in_state_dict = ckpt["state_dict"]
    # in_state_dict = ckpt
    pairings = [
        (src_key, remove_prefix(src_key, "model._orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt["state_dict"] = out_state_dict
    torch.save(ckpt, path)

# %%

if __name__ == "__main__":
    # ckp_path = "/calc/baronig/Projects/sim_results/adlif_rebuttal/compression_task/hydra/2024-12-11/14-42-29/ckpt/last.ckpt"
    # ckp_path = "/calc/baronig/Projects/sim_results/adlif_rebuttal/compression_task/hydra/2025-01-14/16-18-58/ckpt/last.ckpt"
    # ckp_path = "/calc/baronig/Projects/sim_results/adlif_rebuttal/compression_task/hydra/2025-01-14/17-25-47/ckpt/epoch=16-step=66402.ckpt"
    ckp_path = sys.argv[1]
    # repair_checkpoint(ckp_path)
    cfg_path = os.path.join(os.path.dirname(ckp_path), "..", ".hydra", "config.yaml")
    source_waveform, prediction_waveform, spike_probs = create_audio_example(cfg_path, ckp_path, [1, 200, 1529], False)
    # source_waveform, prediction_waveform, spike_probs = create_audio_example(cfg_path, ckp_path, [1], False)

    # %%
    print(f"shape of source_waveform: {torch.tensor(source_waveform[0]).unsqueeze(-1).shape}")
    print(f"shape of prediction_waveform: {torch.tensor(prediction_waveform[0]).unsqueeze(-1).shape}")
    torchaudio.save("source.wav", torch.tensor(source_waveform[0]).unsqueeze(-1), 16000, channels_first=False)
    torchaudio.save("prediction.wav", torch.tensor(prediction_waveform[0]).unsqueeze(-1), 16000, channels_first=False)

    torchaudio.save("source1.wav", torch.tensor(source_waveform[1]).unsqueeze(-1), 16000, channels_first=False)
    torchaudio.save("prediction1.wav", torch.tensor(prediction_waveform[1]).unsqueeze(-1), 16000, channels_first=False)

    torchaudio.save("source2.wav", torch.tensor(source_waveform[2]).unsqueeze(-1), 16000, channels_first=False)
    torchaudio.save("prediction2.wav", torch.tensor(prediction_waveform[2]).unsqueeze(-1), 16000, channels_first=False)
    # %%
    fig = px.line(y=source_waveform[0], title="Source waveform")
    fig.add_trace(go.Scatter(y=prediction_waveform[0], mode='lines', name='Prediction waveform'))
    fig.show()


