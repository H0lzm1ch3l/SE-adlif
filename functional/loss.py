import math
from typing import Any, Union
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from auraloss.utils import apply_reduction
from auraloss.perceptual import FIRFilter
from auraloss.freq import SpectralConvergenceLoss, STFTMagnitudeLoss
reduce_map = {
    'mean': lambda x: torch.mean(x),
    'sum': lambda x: torch.sum(x),
    'none': lambda x: x,
}
class STFTLoss(torch.nn.Module):
    """STFT loss module.

    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).

    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, options include:
           ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        perceptual_weighting (bool, optional): Apply perceptual A-weighting (Sample rate must be supplied). Default: False
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        mag_distance (str, optional): Distance function ["L1", "L2"] for the magnitude loss terms.
        device (str, optional): Place the filterbanks on specified device. Default: None

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'. By default.
        loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance_log: str = "L1",
        mag_distance: str = "L1",
        mel_scale: str = "slaney",
        norm: str = "slaney",
        device: Any = None,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.perceptual_weighting = perceptual_weighting
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance_log = mag_distance_log
        self.mag_distance = mag_distance
        self.mel_scale = mel_scale
        self.norm = norm
        self.device = device
        

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction,
            distance=self.mag_distance_log,
        )
        self.linstft = STFTMagnitudeLoss(
            log=False,
            reduction=reduction,
            distance=self.mag_distance,
        )

        # setup mel filterbank
        if scale is not None:
            try:
                import librosa.filters
            except Exception as e:
                print(e)
                print("Try `pip install auraloss[all]`.")

            if self.scale == "mel":
                assert sample_rate is not None  # Must set sample rate to use mel scale
                assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
                fb = torchaudio.functional.melscale_fbanks(
                    n_freqs=fft_size//2 + 1, f_min=sample_rate/win_length, 
                    f_max=sample_rate/2, 
                    n_mels=n_bins, sample_rate=sample_rate, mel_scale=self.mel_scale,
                    norm=self.norm)

            elif self.scale == "chroma":
                assert sample_rate is not None  # Must set sample rate to use chroma scale
                assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
                fb = librosa.filters.chroma(
                    sr=sample_rate, n_fft=fft_size, n_chroma=n_bins
                )

            else:
                raise ValueError(
                    f"Invalid scale: {self.scale}. Must be 'mel' or 'chroma'."
                )

            self.register_buffer("fb", fb)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

        if self.perceptual_weighting:
            if sample_rate is None:
                raise ValueError(
                    "`sample_rate` must be supplied when `perceptual_weighting = True`."
                )
            self.prefilter = FIRFilter(filter_type="aw", fs=sample_rate)
        # disable complex operation because torch.compile do not support them
        self.stft_call = torch.compiler.disable(self.stft)

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        x_stft =torch.stft(
            input=x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            onesided=True,
        )
        x_mag = torch.clamp(x_stft.abs(), min=self.eps)
        x_phs = torch.angle(x_stft)
        return x_mag, x_phs

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bs, chs, seq_len = input.size()

        if self.perceptual_weighting:  # apply optional A-weighting via FIR filter
            # since FIRFilter only support mono audio we will move channels to batch dim
            input = input.view(bs * chs, 1, -1)
            target = target.view(bs * chs, 1, -1)

            # now apply the filter to both
            self.prefilter.to(input.device)
            input, target = self.prefilter(input, target)

            # now move the channels back
            input = input.view(bs, chs, -1)
            target = target.view(bs, chs, -1)

        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(input.device)
        x_mag, x_phs = self.stft_call(input.view(-1, input.size(-1)))
        y_mag, y_phs = self.stft_call(target.view(-1, target.size(-1)))
        # apply relevant transforms
        if self.scale is not None:
            self.fb = self.fb.to(input.device)
            x_mag = torch.matmul(x_mag.transpose(-1, -2), self.fb).transpose(-1, -2)
            # x_mag = torch.matmul(self.fb, x_mag)
            # y_mag = torch.matmul(self.fb, y_mag)
            y_mag = torch.matmul(y_mag.transpose(-1, -2), self.fb).transpose(-1, -2)
        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1], keepdim=True) / ((y_mag**2).sum([-2, -1], keepdim=True))
            y_mag = y_mag * alpha

        # compute loss terms
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = torch.nn.functional.mse_loss(x_phs, y_phs) if self.w_phs else 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
        )

        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss
class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module wrapper from auraloss
    Expands the interface of auraloss MultiResolutionSTFTLoss 
    with possibility to provide:
        - weights each windows independently
        - mel scaling
        - differents distance (L2, L1) for the spectral, and log_spectral magnitude.
        (Soundstream use L1 for spectral magnitude and L2 for log_spectral)
    

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, list[float], optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, list[float], optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, list[float], optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """
    def __init__(self, fft_sizes: list[int], hop_sizes: list[int], win_lengths: list[int], 
                 window: str = "hann_window",
                 w_sc: Union[float, list[float]] = 1.0,
                 w_log_mag: Union[float, list[float]] = 1.0, 
                 w_lin_mag: Union[float, list[float]] = 0.0, 
                 w_phs: Union[float, list[float]] = 0.0,
                 sample_rate = None, 
                 scale = None, 
                 n_bins = None, 
                 perceptual_weighting = False, 
                 scale_invariance = False,
                 mag_distance_log="L1",
                 mag_distance="L1",
                 mel_scale="slaney",
                 norm="slaney",
                 
                 **kwargs):
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        print(norm)
        num_res = len(fft_sizes)
        if isinstance(w_sc, float):
            w_sc = [w_sc,] * num_res
        if isinstance(w_log_mag, float):
            w_log_mag = [w_log_mag,] * num_res
        if isinstance(w_lin_mag, float):
            w_lin_mag = [w_lin_mag, ] * num_res
        if isinstance(w_phs, float):
            w_phs = [w_phs,] * num_res
            
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl, i_w_sc, i_w_log_mag, i_w_lin_mag, i_w_phs  in zip(fft_sizes, hop_sizes, win_lengths, w_sc, w_log_mag, w_log_mag, w_phs):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    i_w_sc,
                    i_w_log_mag,
                    i_w_lin_mag,
                    i_w_phs,
                    sample_rate,
                    scale,
                    n_bins,
                    perceptual_weighting,
                    scale_invariance,
                    mag_distance=mag_distance,
                    mag_distance_log=mag_distance_log,
                    mel_scale=mel_scale,
                    norm=norm,
                    **kwargs,
                )
            ]
    def forward(self, x, y):

        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for f in self.stft_losses:
            if f.output == "full":  # extract just first term
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)

        mrstft_loss /= len(self.stft_losses)

        if f.output == "loss":
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss



def snn_regularization(spike_proba_per_layer: list[torch.Tensor], target: Union[float | list[float]], 
                       layer_weights: torch.Tensor,
                       reg_type: str, 
                       reduce_neuron: str = "mean",
                       reduce_layer: str = "mean",):
    """
    Determined the regularization loss 

    Args:
        spike_proba_per_layer (torch.Tensor): spike probability for each neuron per layer
        spike_proba_par_layer[i] is a (nums_neuron, ) tensor representing the spike probability 
        for each neuron of layer i 
        target (float): the spike probability target
        reg_type (str): the regularization type
    
    upper regularization: only regularized neurons where the spike probability is higher than the target
    lower regularization: only regularized neurons where the spike probability is lower than the target 
    both: regularized neuron with respect the squared distance of neuron spike probability and the target

    Raises:
        NotImplementedError: Raise error if reg_type is unknown 

    Returns:
        torch.Tensor: the regularization loss 
        dict[str, torch.Tensor] a dictionary of layer_id -> regularization_value for logging 
    """
    if isinstance(target, float):
        target = [target,] * len(spike_proba_per_layer)
        
    target = torch.tensor(target, dtype=spike_proba_per_layer[0].dtype, device=spike_proba_per_layer[0].device)
    
    if reg_type == "lower":
        # reg = [(torch.nn.functional.huber_loss(tg - s) ** 2 ) for tg, s in zip(target, spike_proba_per_layer)]
        reg = [(torch.nn.functional.huber_loss(s, torch.ones_like(s)*tg, reduction="none") * (s < tg).float()) for tg, s in zip(target, spike_proba_per_layer)]

    elif reg_type == "upper":
        reg = [(torch.nn.functional.huber_loss(s, torch.ones_like(s)*tg, reduction="none") * (s > tg).float()) for tg, s in zip(target, spike_proba_per_layer)]
    elif reg_type == "both":
        reg = [((s - tg) ** 2) for tg, s in zip(target, spike_proba_per_layer)]
    else:
        raise NotImplementedError(
            f"Regularization type: {reg_type} is not implemented, valid type are [upper, lower, both]"
        )
    if reduce_neuron == 'mean':
        reg = torch.stack([r.mean() for r in reg])
    elif reduce_neuron == 'sum':
        reg = torch.stack([r.sum() for r in reg])
    res = (layer_weights * reg)
    log_dict = {f"{reg_type}_layer_{k}": res[k] for k in range(len(res))}
    if reduce_layer == 'mean':
        return res.mean(), log_dict
    elif reduce_layer == 'sum':
        return res.sum(), log_dict
    
def calculate_weight_decay_loss(model):
    weight_decay_loss = 0
    for name, param in model.named_parameters():
        if (
            "bias" not in name
            and "tau" not in name
            and "beta" not in name
            and "coeff" not in name
        ):
            weight_decay_loss += torch.mean(param**2)
    return weight_decay_loss
