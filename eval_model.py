
import torch
from datasets.audio_compress import LibriTTS
from datasets.utils.pad_tensors import PadTensors
from torch.utils.data import DataLoader
import argparse
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
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
def get_decay(dt: float, tau_min:Array, tau_max: Array, w: Array):
    coef = jax.nn.sigmoid(w)
    alpha_min = jnp.exp(-dt/tau_min)
    alpha_max = jnp.exp(-dt/tau_max)
    return alpha_max * coef + (1.0  - coef)*alpha_min
def adlif_pytorch_states_dict_to_jax_dict(ckpt_path):
    ckpt = torch.load(ckpt_path)
    ckpt_state_dict = ckpt['state_dict']
    ckpt_state_dict = repair_checkpoint(ckpt_state_dict)
    q = ckpt['hyper_parameters']['cfg']['encoder']['l1']['q']
    nb_bottleneck_neurons = ckpt['hyper_parameters']['cfg']['bottleneck_neurons']
    thr = ckpt['hyper_parameters']['cfg']['encoder']['l1']['thr']
    ckpt_state_dict = jax.tree.map(lambda x: jnp.array(x.cpu().numpy(), dtype=jnp.float32), ckpt_state_dict)
    ckpt_state_dict.update({"q": q, "nb_bottleneck_neurons": nb_bottleneck_neurons,"thr": thr})
    return ckpt_state_dict

def se_adlif_model(weights: dict[str, Array], inputs:Array):
    enc_l1_u0 = weights['model.encoder.l1.u0']
    enc_l1_w0 = jnp.zeros_like(enc_l1_u0)
    
    enc_l2_u0 = weights['model.encoder.l2.u0']
    enc_l2_w0 = jnp.zeros_like(enc_l2_u0)
    enc_l2_z0 = jnp.zeros_like(enc_l2_u0)
    
    dec_l1_u0 = weights['model.decoder.l1.u0']
    dec_l1_w0 = jnp.zeros_like(dec_l1_u0)
    dec_l1_z0 = jnp.zeros_like(dec_l1_u0)

    dec_l2_u0 = weights['model.decoder.l2.u0']
    dec_l2_w0 = jnp.zeros_like(dec_l2_u0)
    dec_l2_z0 = jnp.zeros_like(dec_l2_u0)


    lout_u = jnp.zeros((256,), dtype=jnp.float32)
    
    enc_l1_decay_u = get_decay(1.0, 
                               weights['model.encoder.l1.tau_u_trainer.tau_min'], 
                               weights['model.encoder.l1.tau_u_trainer.tau_max'],
                               weights['model.encoder.l1.tau_u_trainer.weight'])
    enc_l1_decay_w = get_decay(1.0,
                               weights['model.encoder.l1.tau_w_trainer.tau_min'], 
                               weights['model.encoder.l1.tau_w_trainer.tau_max'], 
                               weights['model.encoder.l1.tau_w_trainer.weight'])

    enc_l2_decay_u = get_decay(1.0, 
                               weights['model.encoder.l2.tau_u_trainer.tau_min'], 
                               weights['model.encoder.l2.tau_u_trainer.tau_max'],
                               weights['model.encoder.l2.tau_u_trainer.weight'])
    enc_l2_decay_w = get_decay(1.0, 
                               weights['model.encoder.l2.tau_w_trainer.tau_min'], 
                               weights['model.encoder.l2.tau_w_trainer.tau_max'], 
                               weights['model.encoder.l2.tau_w_trainer.weight'])

    dec_l1_decay_u = get_decay(1.0, 
                               weights['model.decoder.l1.tau_u_trainer.tau_min'], 
                               weights['model.decoder.l1.tau_u_trainer.tau_max'], 
                               weights['model.decoder.l1.tau_u_trainer.weight'])
    dec_l1_decay_w = get_decay(1.0, 
                               weights['model.decoder.l1.tau_w_trainer.tau_min'], 
                               weights['model.decoder.l1.tau_w_trainer.tau_max'], 
                               weights['model.decoder.l1.tau_w_trainer.weight'])

    dec_l2_decay_u = get_decay(1.0, 
                               weights['model.decoder.l2.tau_u_trainer.tau_min'], 
                               weights['model.decoder.l2.tau_u_trainer.tau_max'],
                               weights['model.decoder.l2.tau_u_trainer.weight'])
    dec_l2_decay_w = get_decay(1.0, 
                               weights['model.decoder.l2.tau_w_trainer.tau_min'], 
                               weights['model.decoder.l2.tau_w_trainer.tau_max'], 
                               weights['model.decoder.l2.tau_w_trainer.weight'])
    
    lout_decay = get_decay(1.0, 
                           weights['model.decoder.out_layer.tau_u_trainer.tau_min'], 
                           weights['model.decoder.out_layer.tau_u_trainer.tau_max'], 
                           weights['model.decoder.out_layer.tau_u_trainer.weight'])
    
    enc_l1_a = weights['model.encoder.l1.a']
    enc_l1_b = weights['model.encoder.l1.b']
    
    enc_l2_a = weights['model.encoder.l2.a']
    enc_l2_b = weights['model.encoder.l2.b']
    
    dec_l1_a = weights['model.decoder.l1.a']
    dec_l1_b = weights['model.decoder.l2.b']

    dec_l2_a = weights['model.decoder.l2.a']
    dec_l2_b = weights['model.decoder.l2.b']
    
    q = weights['q']
    nb_bottleneck_neurons = weights['nb_bottleneck_neurons']
    def loop(carry, inputs):
        enc_l1_u, enc_l1_w, enc_l2_u, enc_l2_w, enc_l2_z, dec_l1_u, dec_l1_w, dec_l1_z, dec_l2_u, dec_l2_w, dec_l2_z, lout_u = carry
        cur = inputs @ weights['model.encoder.l1.weight'].T + weights['model.encoder.l1.bias']
        enc_l1_u = enc_l1_decay_u * enc_l1_u + (1.0 - enc_l1_decay_u) * (cur - enc_l1_w)
        enc_l1_z = jnp.heaviside(enc_l1_u - weights['thr'], 0.0)
        enc_l1_u = enc_l1_u * (1 - enc_l1_z) + enc_l1_u0*enc_l1_z
        enc_l1_w = enc_l1_decay_w * enc_l1_w + (1.0 - enc_l1_decay_w) * (enc_l1_a * enc_l1_u + enc_l1_b * enc_l1_z) * q
        
        cur = enc_l1_z @ weights['model.encoder.l2.weight'].T  + enc_l2_z @ weights['model.encoder.l2.recurrent'].T + weights['model.encoder.l2.bias']
        enc_l2_u = enc_l2_decay_u * enc_l2_u + (1.0 - enc_l2_decay_u) * (cur - enc_l2_w)
        enc_l2_z = jnp.heaviside(enc_l2_u - weights['thr'], 0.0)
        enc_l2_u = enc_l2_u * (1 - enc_l2_z) + enc_l2_u0*enc_l2_z
        enc_l2_w = enc_l2_decay_w * enc_l2_w + (1.0 - enc_l2_decay_w) * (enc_l2_a * enc_l2_u + enc_l2_b * enc_l2_z) * q
        
        cur = enc_l2_z[: nb_bottleneck_neurons] @ weights['model.decoder.l1.weight'].T  + dec_l1_z @ weights['model.decoder.l1.recurrent'].T + weights['model.decoder.l1.bias']
        dec_l1_u = dec_l1_decay_u * dec_l1_u + (1.0 - dec_l1_decay_u) * (cur - dec_l1_w)
        dec_l1_z = jnp.heaviside(dec_l1_u - weights['thr'], 0.0)
        dec_l1_u = dec_l1_u * (1 - dec_l1_z) + dec_l1_u0*dec_l1_z
        dec_l1_w = dec_l1_decay_w * dec_l1_w + (1.0 - dec_l1_decay_w) * (dec_l1_a * dec_l1_u + dec_l1_b * dec_l1_z) * q
        
        cur = dec_l1_z @ weights['model.decoder.l2.weight'].T  + dec_l2_z @ weights['model.decoder.l2.recurrent'].T + weights['model.decoder.l2.bias']
        dec_l2_u = dec_l2_decay_u * dec_l2_u + (1.0 - dec_l2_decay_u) * (cur - dec_l2_w)
        dec_l2_z = jnp.heaviside(dec_l2_u - weights['thr'], 0.0)
        dec_l2_u = dec_l2_u * (1 - dec_l2_z) + dec_l2_u0*dec_l2_z
        dec_l2_w = dec_l2_decay_w * dec_l2_w + (1.0 - dec_l2_decay_w) * (dec_l2_a * dec_l2_u + dec_l2_b * dec_l2_z) * q
        
        cur = dec_l2_z @ weights['model.decoder.out_layer.weight'].T   + weights['model.decoder.out_layer.bias']
        lout_u = lout_decay * lout_u + (1.0 - lout_decay) * cur
        return (enc_l1_u, enc_l1_w, enc_l2_u, enc_l2_w, enc_l2_z, dec_l1_u, dec_l1_w, dec_l1_z, dec_l2_u, dec_l2_w, dec_l2_z, lout_u), lout_u
    _, out = jax.lax.scan(loop, (enc_l1_u0, enc_l1_w0, enc_l2_u0, enc_l2_w0, enc_l2_z0, dec_l1_u0, dec_l1_w0, dec_l1_z0, dec_l2_u0, dec_l2_w0, dec_l2_z0, lout_u),
                 inputs)
    probs = jax.nn.softmax(out/weights['loss.temp'], -1)
    out = inverse_A_law(jnp.sum(probs * weights['loss.bin_edges'][None, None,...], axis=-1, keepdims=True))
    return out
    
def inverse_A_law(y: Array, a: float = 87.6):
    sign_y = jnp.sign(y)
    abs_y = jnp.abs(y)
    log_a_p1 = jnp.log(a) + 1
    x1 = (abs_y*log_a_p1)/a
    x2 = jnp.exp(-1 + abs_y*log_a_p1)/a
    x = jnp.where(abs_y < 1/log_a_p1, x1, x2)    
    return sign_y*x

def adlif_dataloder_loop(weights, py_dataloader):
    model = jax.vmap(partial(se_adlif_model, weights))
    for batch in tqdm(py_dataloader, "evaluation"):
        inputs, targets, block_idx = batch
        out = model(jnp.array(inputs, dtype=jnp.float32))

def create_test_libri_dataloader(ckpt_path, batch_size, num_workers):
    ckpt = torch.load(ckpt_path)
    def delay_transform(inputs, targets, block_idx):
            # add zero padding to account for possible prediciton delay
            # idea is that it is potentially complex for the model to predict
            # y[t] = L(x[0:t]) where L is the model, x inputs, y the output
            # delay allow predition as y[t - delay] = L([x[0:t]])
            inputs = torch.concatenate(
                (
                    inputs,
                    torch.zeros(
                        (
                            ckpt['hyper_parameters']['cfg']['dataset']['prediction_delay'], 1), 
                        device=inputs.device, dtype=inputs.dtype
                    ),
                )
            )
            block_idx = torch.ones((inputs.shape[0],), dtype=torch.int32)
            return inputs, targets, block_idx
    
    dataset = LibriTTS(ckpt['hyper_parameters']['cfg']['datadir'], ckpt['hyper_parameters']['cfg']['cachedir'], 
                    sampling_freq=16_000, sample_length=-1, 
                    normalization=ckpt['hyper_parameters']['cfg']['dataset']['normalization'],
                    debug=False,
                    train=False, 
                    full_transform=delay_transform)
    collate_fn = PadTensors(require_padding=True)
    
    data_loader = DataLoader(dataset,
                             shuffle=False,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False,
                             collate_fn=collate_fn,
                             num_workers=num_workers,
                             persistent_workers=num_workers >0
                             )
    return data_loader
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='eval_metrics')
    parser.add_argument('--ckpt_path', help='path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    py_dataloader = create_test_libri_dataloader(args.ckpt_path, args.batch_size, 0)
    weight = adlif_pytorch_states_dict_to_jax_dict(args.ckpt)
    adlif_dataloder_loop(weight, py_dataloader)