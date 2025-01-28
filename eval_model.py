
from pathlib import Path
import hydra
from omegaconf import DictConfig

import torch
import torchaudio
from functools import partial

from tqdm import tqdm

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return 'model.' + text[len(prefix) :]
    return text
def repair_checkpoint(in_state_dict):
    pairings = [
        (src_key, remove_prefix(src_key, "model._orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  in_state_dict
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = in_state_dict[src_key]
    return out_state_dict
def get_decay(dt: float, alpha_min, alpha_max, w):
    import jax
    coef = jax.nn.sigmoid(w)
    return alpha_max * coef + (1.0  - coef)*alpha_min 

def adlif_pytorch_states_dict_to_jax_dict(ckpt_path):
    import jax
    import jax.numpy as jnp
    ckpt = torch.load(ckpt_path, map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    ckpt_state_dict = ckpt['state_dict']
    ckpt_state_dict = repair_checkpoint(ckpt_state_dict)
    q = ckpt['hyper_parameters']['cfg']['encoder']['l1']['q']
    nb_bottleneck_neurons = ckpt['hyper_parameters']['cfg']['bottleneck_neurons']
    ckpt_state_dict = jax.tree.map(lambda x: jnp.array(x.cpu().numpy(), dtype=jnp.float32), ckpt_state_dict)
    enc_l1_decay_u = get_decay(1.0, 
                               ckpt_state_dict['model.encoder.l1.tau_u_trainer.alpha_min'], 
                               ckpt_state_dict['model.encoder.l1.tau_u_trainer.alpha_max'],
                               ckpt_state_dict['model.encoder.l1.tau_u_trainer.weight'])


    enc_l2_decay_u = get_decay(1.0, 
                               ckpt_state_dict['model.encoder.l2.tau_u_trainer.alpha_min'], 
                               ckpt_state_dict['model.encoder.l2.tau_u_trainer.alpha_max'],
                               ckpt_state_dict['model.encoder.l2.tau_u_trainer.weight'])


    dec_l1_decay_u = get_decay(1.0, 
                               ckpt_state_dict['model.decoder.l1.tau_u_trainer.alpha_min'], 
                               ckpt_state_dict['model.decoder.l1.tau_u_trainer.alpha_max'], 
                               ckpt_state_dict['model.decoder.l1.tau_u_trainer.weight'])


    dec_l2_decay_u = get_decay(1.0, 
                               ckpt_state_dict['model.decoder.l2.tau_u_trainer.alpha_min'], 
                               ckpt_state_dict['model.decoder.l2.tau_u_trainer.alpha_max'],
                               ckpt_state_dict['model.decoder.l2.tau_u_trainer.weight'])

    
    lout_decay = get_decay(1.0, 
                           ckpt_state_dict['model.decoder.out_layer.tau_u_trainer.alpha_min'], 
                           ckpt_state_dict['model.decoder.out_layer.tau_u_trainer.alpha_max'], 
                           ckpt_state_dict['model.decoder.out_layer.tau_u_trainer.weight'])
    # print(ckpt['hyper_parameters']['cfg']['cell'])
    # if ckpt['hyper_parameters']['cfg']['cell'] != 'lif':
    
    enc_l1_decay_w = get_decay(1.0,
                            ckpt_state_dict['model.encoder.l1.tau_w_trainer.alpha_min'], 
                            ckpt_state_dict['model.encoder.l1.tau_w_trainer.alpha_max'], 
                            ckpt_state_dict['model.encoder.l1.tau_w_trainer.weight'])
    
    enc_l2_decay_w = get_decay(1.0, 
                            ckpt_state_dict['model.encoder.l2.tau_w_trainer.alpha_min'], 
                            ckpt_state_dict['model.encoder.l2.tau_w_trainer.alpha_max'], 
                            ckpt_state_dict['model.encoder.l2.tau_w_trainer.weight'])

    dec_l1_decay_w = get_decay(1.0, 
                            ckpt_state_dict['model.decoder.l1.tau_w_trainer.alpha_min'], 
                            ckpt_state_dict['model.decoder.l1.tau_w_trainer.alpha_max'], 
                            ckpt_state_dict['model.decoder.l1.tau_w_trainer.weight'])
    
    dec_l2_decay_w = get_decay(1.0, 
                            ckpt_state_dict['model.decoder.l2.tau_w_trainer.alpha_min'], 
                            ckpt_state_dict['model.decoder.l2.tau_w_trainer.alpha_max'], 
                            ckpt_state_dict['model.decoder.l2.tau_w_trainer.weight'])
    # else:
    #     zero = jnp.zeros(())
    #     enc_l1_decay_w = enc_l2_decay_w = dec_l1_decay_w = dec_l2_decay_w = zero
    #     ckpt_state_dict['model.encoder.l1.a'] = zero
    #     ckpt_state_dict['model.encoder.l1.b'] = zero
        
    #     ckpt_state_dict['model.encoder.l2.a'] = zero
    #     ckpt_state_dict['model.encoder.l2.b'] = zero
        
    #     ckpt_state_dict['model.decoder.l1.a'] = zero
    #     ckpt_state_dict['model.decoder.l1.b'] = zero

    #     ckpt_state_dict['model.decoder.l2.a'] = zero
    #     ckpt_state_dict['model.decoder.l2.b'] = zero
    transition_begin = ckpt['hyper_parameters']['cfg']['loss']['transition_begin']
    transition_steps = ckpt['hyper_parameters']['cfg']['loss']['transition_steps']
    temp_decay = ckpt['hyper_parameters']['cfg']['loss']['temp_decay']
    factor = (ckpt_state_dict['loss.batch_count'] - transition_begin) / transition_steps
    min_temp = ckpt_state_dict['loss.min_temp']
    temp = ckpt_state_dict['loss.temp']
    ckpt_state_dict['loss.temp_'] = jnp.maximum(min_temp, temp*temp_decay**factor)
    print(ckpt_state_dict['loss.temp_'])
    ckpt_state_dict.update(
        {
            "model.encoder.l1.decay_u": enc_l1_decay_u,
            "model.encoder.l1.decay_w": enc_l1_decay_w,
            "model.encoder.l2.decay_u": enc_l2_decay_u,
            "model.encoder.l2.decay_w": enc_l2_decay_w,
            "model.decoder.l1.decay_u": dec_l1_decay_u,
            "model.decoder.l1.decay_w": dec_l1_decay_w,
            "model.decoder.l2.decay_u": dec_l2_decay_u,
            "model.decoder.l2.decay_w": dec_l2_decay_w,
            "model.decoder.out_layer.decay": lout_decay,
            "q": q, "nb_bottleneck_neurons": nb_bottleneck_neurons,
            }
        )
    return ckpt_state_dict
def se_adlif_model(model, weights: dict, inputs):
    import jax
    import jax.numpy as jnp
    enc_l1_u0 = weights['model.encoder.l1.u0']
    enc_l1_w0 = jnp.zeros_like(enc_l1_u0)
    enc_l1_z0 = jnp.zeros_like(enc_l1_u0)
    
    enc_l2_u0 = weights['model.encoder.l2.u0']
    enc_l2_w0 = jnp.zeros_like(enc_l2_u0)
    enc_l2_z0 = jnp.zeros_like(enc_l2_u0)
    
    dec_l1_u0 = weights['model.decoder.l1.u0']
    dec_l1_w0 = jnp.zeros_like(dec_l1_u0)
    dec_l1_z0 = jnp.zeros_like(dec_l1_u0)

    dec_l2_u0 = weights['model.decoder.l2.u0']
    dec_l2_w0 = jnp.zeros_like(dec_l2_u0)
    dec_l2_z0 = jnp.zeros_like(dec_l2_u0)


    lout_u = jnp.zeros_like(weights['model.decoder.out_layer.decay'], dtype=jnp.float32)
    

    
    enc_l1_a = weights['model.encoder.l1.a']
    enc_l1_b = weights['model.encoder.l1.b']
    
    enc_l2_a = weights['model.encoder.l2.a']
    enc_l2_b = weights['model.encoder.l2.b']
    
    dec_l1_a = weights['model.decoder.l1.a']
    dec_l1_b = weights['model.decoder.l1.b']

    dec_l2_a = weights['model.decoder.l2.a']
    dec_l2_b = weights['model.decoder.l2.b']
    
    enc_l1_decay_u = weights["model.encoder.l1.decay_u"]    
    enc_l1_decay_w = weights["model.encoder.l1.decay_w"]
    
    enc_l2_decay_u = weights["model.encoder.l2.decay_u"]
    enc_l2_decay_w = weights["model.encoder.l2.decay_w"]
    
    dec_l1_decay_u = weights["model.decoder.l1.decay_u"]
    dec_l1_decay_w = weights["model.decoder.l1.decay_w"]
    
    dec_l2_decay_u = weights["model.decoder.l2.decay_u"]
    dec_l2_decay_w = weights["model.decoder.l2.decay_w"]
    lout_decay = weights["model.decoder.out_layer.decay"]
    
    q = weights['q']
    nb_bottleneck_neurons = weights['nb_bottleneck_neurons']        
    print(inputs.shape)
    def loop(carry, inputs):
        enc_l1_u, enc_l1_w, enc_l2_u, enc_l2_w, enc_l2_z, dec_l1_u, dec_l1_w, dec_l1_z, dec_l2_u, dec_l2_w, dec_l2_z, lout_u = carry
        cur = inputs @ weights['model.encoder.l1.weight'].T + weights['model.encoder.l1.bias']
        enc_l1_u = enc_l1_decay_u * enc_l1_u + (1.0 - enc_l1_decay_u) * (cur - enc_l1_w)
        enc_l1_z = jnp.heaviside(enc_l1_u - weights['model.encoder.l1.thr'], 0.0)
        enc_l1_u = enc_l1_u * (1 - enc_l1_z) + enc_l1_u0*enc_l1_z
        enc_l1_w = enc_l1_decay_w * enc_l1_w + (1.0 - enc_l1_decay_w) * (enc_l1_a * enc_l1_u + enc_l1_b * enc_l1_z) * q
        
        cur = enc_l1_z @ weights['model.encoder.l2.weight'].T  + enc_l2_z @ weights['model.encoder.l2.recurrent'].T + weights['model.encoder.l2.bias']
        enc_l2_u = enc_l2_decay_u * enc_l2_u + (1.0 - enc_l2_decay_u) * (cur - enc_l2_w)
        enc_l2_z = jnp.heaviside(enc_l2_u - weights['model.encoder.l2.thr'], 0.0)
        enc_l2_u = enc_l2_u * (1 - enc_l2_z) + enc_l2_u0*enc_l2_z
        enc_l2_w = enc_l2_decay_w * enc_l2_w + (1.0 - enc_l2_decay_w) * (enc_l2_a * enc_l2_u + enc_l2_b * enc_l2_z) * q
        
        cur = enc_l2_z[: nb_bottleneck_neurons] @ weights['model.decoder.l1.weight'].T  + dec_l1_z @ weights['model.decoder.l1.recurrent'].T + weights['model.decoder.l1.bias']
        dec_l1_u = dec_l1_decay_u * dec_l1_u + (1.0 - dec_l1_decay_u) * (cur - dec_l1_w)
        dec_l1_z = jnp.heaviside(dec_l1_u - weights['model.decoder.l1.thr'], 0.0)
        dec_l1_u = dec_l1_u * (1 - dec_l1_z) + dec_l1_u0*dec_l1_z
        dec_l1_w = dec_l1_decay_w * dec_l1_w + (1.0 - dec_l1_decay_w) * (dec_l1_a * dec_l1_u + dec_l1_b * dec_l1_z) * q
        
        cur = dec_l1_z @ weights['model.decoder.l2.weight'].T  + dec_l2_z @ weights['model.decoder.l2.recurrent'].T + weights['model.decoder.l2.bias']
        dec_l2_u = dec_l2_decay_u * dec_l2_u + (1.0 - dec_l2_decay_u) * (cur - dec_l2_w)
        dec_l2_z = jnp.heaviside(dec_l2_u - weights['model.decoder.l2.thr'], 0.0)
        dec_l2_u = dec_l2_u * (1 - dec_l2_z) + dec_l2_u0*dec_l2_z
        dec_l2_w = dec_l2_decay_w * dec_l2_w + (1.0 - dec_l2_decay_w) * (dec_l2_a * dec_l2_u + dec_l2_b * dec_l2_z) * q
        
        cur = dec_l2_z @ weights['model.decoder.out_layer.weight'].T + weights['model.decoder.out_layer.bias']
        lout_u = lout_decay * lout_u + (1.0 - lout_decay) * cur
        probs = jax.nn.softmax(lout_u/weights['loss.temp_'], -1)
        out = inverse_A_law(jnp.sum(probs * weights['loss.bin_edges'], keepdims=True))
        return (enc_l1_u, enc_l1_w, 
                enc_l2_u, enc_l2_w, enc_l2_z, 
                dec_l1_u, dec_l1_w, dec_l1_z, 
                dec_l2_u, dec_l2_w, dec_l2_z, 
                lout_u), out
    _, out = jax.lax.scan(loop, 
                          (enc_l1_u0, enc_l1_w0, 
                           enc_l2_u0, enc_l2_w0, enc_l2_z0, 
                           dec_l1_u0, dec_l1_w0, dec_l1_z0, 
                           dec_l2_u0, dec_l2_w0, dec_l2_z0, 
                           lout_u),
                 inputs, unroll=1)
    return out
    
    
def inverse_A_law(y, a: float = 87.6):
    import jax.numpy as jnp
    sign_y = jnp.sign(y)
    abs_y = jnp.abs(y)
    log_a_p1 = jnp.log(a) + 1
    x1 = (abs_y*log_a_p1)/a
    x2 = jnp.exp(-1 + abs_y*log_a_p1)/a
    x = jnp.where(abs_y < 1/log_a_p1, x1, x2)    
    return sign_y*x

def adlif_dataloder_loop(model, weights, prediction_delay, wave_list_sorted, save_to):
    import jax
    import jax.numpy as jnp
    save_to = Path(save_to)
    save_to.mkdir(exist_ok=True, parents=True)
    model = partial(se_adlif_model, model, weights)
    model = jax.jit(jax.vmap(model),)
    i = 0
    for wave in tqdm(wave_list_sorted, "evaluation"):
        _inputs = torchaudio.load(wave)[0].T.unsqueeze(0)
        inputs = jnp.array(_inputs.cpu().numpy(), dtype=jnp.float32,)
        inputs = jnp.concat((inputs, jnp.zeros((inputs.shape[0], prediction_delay, inputs.shape[2]), dtype=inputs.dtype)), axis=1)
        out = model(inputs)
        jnp.save(save_to/str(i), out[:, prediction_delay:], allow_pickle=False)
        i += 1

def create_test_libri_dataloader(ckpt_path, test_dataset_dir):
    ckpt = torch.load(ckpt_path, map_location=None if torch.cuda.is_available() else torch.device('cpu'))

    # glob every waveform the dataset_dir sort them return the list of file
    wave_list_sorted = sorted(list(Path(test_dataset_dir).rglob("*.wav")))
    
    
    return ckpt['hyper_parameters']['cfg']['dataset']['prediction_delay'], wave_list_sorted
@hydra.main(config_path="config", config_name="eval_run", version_base=None)
def main(cfg: DictConfig):
    import jax
    import jax.numpy as jnp
    prediction_delay, wave_list_sorted = create_test_libri_dataloader(cfg.ckpt_path, cfg.test_dir_path)
    weight = adlif_pytorch_states_dict_to_jax_dict(cfg.ckpt_path)
    adlif_dataloder_loop(cfg.model, weight, prediction_delay, wave_list_sorted, cfg.save_to)

if __name__ == "__main__":
    main()
