"""
"""

from decimal import localcontext, Decimal, ROUND_HALF_UP
from typing import Optional, Sequence, Union

import pywt
import numpy as np
from easydict import EasyDict as ED

import numpy as np
import scipy.signal as SS
import plotly.express as px
from easydict import EasyDict as ED
from torch_ecg.utils.utils_signal import butter_bandpass_filter, normalize




__all__ = [
    "schmidt_spike_removal",
    "get_springer_features",
    "hilbert_envelope",
    "homomorphic_envelope_with_hilbert",
    "get_PSD_feature",
    "get_springer_features",
    "get_dwt_features",
    "get_full_dwt_features",
]


def schmidt_spike_removal(
    original_signal: np.ndarray,
    fs: int,
    window_size: float = 0.5,
    threshold: float = 3.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    spike removal using Schmidt algorithm

    Parameters
    ----------
    original_signal : np.ndarray,
        the original signal
    fs : int,
        the sampling frequency
    window_size : float, default 0.5,
        the sliding window size, with units in seconds
    threshold : float, default 3.0,
        the threshold (multiplier for the median value) for detecting spikes
    eps : float, default 1e-4,
        the epsilon for numerical stability

    Returns
    -------
    despiked_signal : np.ndarray,
        the despiked signal

    """
    window_size = round(fs * window_size)
    nframes, res = divmod(original_signal.shape[0], window_size)
    frames = original_signal[: window_size * nframes].reshape((nframes, window_size))
    if res > 0:
        nframes += 1
        frames = np.concatenate(
            (frames, original_signal[-window_size:].reshape((1, window_size))), axis=0
        )
    MAAs = np.abs(frames).max(axis=1)  # of shape (nframes,)

    while len(np.where(MAAs > threshold * np.median(MAAs))[0]) > 0:
        frame_num = np.where(MAAs == MAAs.max())[0][0]
        spike_position = np.argmax(np.abs(frames[frame_num]))
        zero_crossings = np.where(np.diff(np.sign(frames[frame_num])))[0]
        spike_start = np.where(zero_crossings <= spike_position)[0]
        spike_start = zero_crossings[spike_start[-1]] if len(spike_start) > 0 else 0
        spike_end = np.where(zero_crossings >= spike_position)[0]
        spike_end = (
            zero_crossings[spike_end[0]] + 1 if len(spike_end) > 0 else window_size
        )
        frames[frame_num, spike_start:spike_end] = eps
        MAAs = np.abs(frames).max(axis=1)

    despiked_signal = original_signal.copy()
    if res > 0:
        despiked_signal[-window_size:] = frames[-1]
        nframes -= 1
    despiked_signal[: window_size * nframes] = frames[:nframes, ...].reshape((-1,))

    return despiked_signal



def get_springer_features(
    signal: np.ndarray,
    fs: int,
    feature_fs: int,
    feature_format: str = "flat",
    config: Optional[dict] = None,
) -> np.ndarray:
    """
    This function **almost** re-implements the original matlab
    implementation of the Springer features.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.
    feature_fs: int,
        The sampling frequency of the features.
    feature_format: str, default "flat",
        The format of the features, can be one of
        "flat", "channel_first", "channel_last",
        case insensitive.
    config: dict, optional,
        The configuration for extraction methods of the features.

    Returns
    -------
    springer_features: np.ndarray,
        The extracted features, of shape
        (4 * feature_len,) if `feature_format` is "flat",
        (4, feature_len) if `feature_format` is "channel_first",
        (feature_len, 4) if `feature_format` is "channel_last".
        The features are in the following order:
        - homomorphic_envelope
        - hilbert envelope
        - PSD
        - DWT

    """
    assert feature_format.lower() in [
        "flat",
        "channel_first",
        "channel_last",
    ], f"`feature_format` must be one of 'flat', 'channel_first', 'channel_last', but got {feature_format}"
    cfg = ED(
        order=2,
        lowcut=25,
        highcut=400,
        lpf_freq=8,
        seg_tol=0.1,
        psd_freq_lim=(40, 60),
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    filtered_signal = butter_bandpass_filter(
        signal,
        fs=fs,
        lowcut=cfg.lowcut,
        highcut=cfg.highcut,
        order=cfg.order,
        btype="lohi",
    )
    filtered_signal = schmidt_spike_removal(filtered_signal, fs)

    homomorphic_envelope = homomorphic_envelope_with_hilbert(
        filtered_signal, fs, lpf_freq=cfg.lpf_freq
    )
    downsampled_homomorphic_envelope = SS.resample_poly(
        homomorphic_envelope, feature_fs, fs
    )
    downsampled_homomorphic_envelope = normalize(
        downsampled_homomorphic_envelope, method="z-score", mean=0.0, std=1.0
    )

    amplitude_envelope = hilbert_envelope(filtered_signal, fs)
    downsampled_hilbert_envelope = SS.resample_poly(amplitude_envelope, feature_fs, fs)
    downsampled_hilbert_envelope = normalize(
        downsampled_hilbert_envelope, method="z-score", mean=0.0, std=1.0
    )

    psd = get_PSD_feature(filtered_signal, fs, freq_lim=cfg.psd_freq_lim)
    psd = SS.resample_poly(psd, len(downsampled_homomorphic_envelope), len(psd))
    psd = normalize(psd, method="z-score", mean=0.0, std=1.0)

    wavelet_feature = np.abs(get_dwt_features(filtered_signal, fs, config=cfg))
    wavelet_feature = wavelet_feature[: len(homomorphic_envelope)]
    wavelet_feature = SS.resample_poly(wavelet_feature, feature_fs, fs)
    wavelet_feature = normalize(wavelet_feature, method="z-score", mean=0.0, std=1.0)

    func = dict(
        flat=np.concatenate,
        channel_first=np.row_stack,
        channel_last=np.column_stack,
    )

    springer_features = func[feature_format.lower()](
        [
            downsampled_homomorphic_envelope,
            downsampled_hilbert_envelope,
            psd,
            wavelet_feature,
        ]
    )
    return springer_features


def hilbert_envelope(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute the envelope of the signal using the Hilbert transform.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.

    Returns
    -------
    ndarray:
        The envelope of the signal.

    """
    return np.abs(SS.hilbert(signal))


def homomorphic_envelope_with_hilbert(
    signal: np.ndarray, fs: int, lpf_freq: int = 8, order: int = 1
) -> np.ndarray:
    """
    Compute the homomorphic envelope of the signal using the Hilbert transform.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.
    lpf_freq: int, default 8,
        The low-pass filter frequency (high cut frequency).
        The filter will be applied to log of the Hilbert envelope.
    order: int, default 1,
        The order of the butterworth low-pass filter.

    Returns
    -------
    homomorphic_envelope: ndarray,
        The homomorphic envelope of the signal.

    """
    amplitude_envelope = hilbert_envelope(signal, fs)
    homomorphic_envelope = np.exp(
        butter_bandpass_filter(np.log(amplitude_envelope), 0, lpf_freq, fs, order=order)
    )
    homomorphic_envelope[0] = homomorphic_envelope[1]
    return homomorphic_envelope


def get_PSD_feature(
    signal: np.ndarray,
    fs: int,
    freq_lim: Sequence[int] = (40, 60),
    window_size: float = 1 / 40,
    overlap_size: float = 1 / 80,
) -> np.ndarray:
    """
    Compute the PSD (power spectral density) of the signal.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.
    freq_lim: sequence of int, default (40,60),
        The frequency range to compute the PSD.
    window_size: float, default 1/40,
        The size of the window to compute the PSD,
        with units in seconds.
    overlap_size: float, default 1/80,
        The size of the overlap between windows to compute the PSD,
        with units in seconds.

    Returns
    -------
    psd: ndarray,
        The PSD of the signal.

    NOTE:
    The `round` function in matlab is different from python's `round` function,
    ref. https://en.wikipedia.org/wiki/IEEE_754#Rounding_rules.
    The rounding rule for matlab is `to nearest, ties away from zero`,
    while the rounding rule for python is `to nearest, ties to even`.

    """
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        nperseg = int(Decimal(fs * window_size).to_integral_value())
        noverlap = int(Decimal(fs * overlap_size).to_integral_value())
    f, t, Sxx = SS.spectrogram(
        signal,
        fs,
        "hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=fs,
        return_onesided=True,
        scaling="density",
        mode="psd",
    )
    inds = np.where((f >= freq_lim[0]) & (f <= freq_lim[1]))[0]
    psd = np.mean(Sxx[inds, :], axis=0)
    return psd

def get_dwt_features(
    signal: np.ndarray, fs: int, config: Optional[dict] = None
) -> np.ndarray:
    """
    compute the discrete wavelet transform (DWT) features using Springer's algorithm

    Parameters
    ----------
    signal : np.ndarray,
        the (PCG) signal, of shape (nsamples,)
    fs : int,
        the sampling frequency
    config : dict, optional,
        the configuration, with the following keys:
        - ``'wavelet_level'``: int,
            the level of the wavelet decomposition, default: 3
        - ``'wavelet_name'``: str,
            the name of the wavelet, default: "db7"

    Returns
    -------
    dwt_features : np.ndarray,
        the DWT features, of shape (nsamples,)

    """
    cfg = ED(
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    siglen = len(signal)

    detail_coefs = pywt.downcoef(
        "d", signal, wavelet=cfg.wavelet_name, level=cfg.wavelet_level
    )
    dwt_features = _wkeep1(np.repeat(detail_coefs, 2**cfg.wavelet_level), siglen)
    return dwt_features


def get_full_dwt_features(
    signal: np.ndarray, fs: int, config: Optional[dict] = None
) -> np.ndarray:
    """
    compute the full DWT features using Springer's algorithm

    Parameters
    ----------
    signal : np.ndarray,
        the (PCG) signal, of shape (nsamples,)
    fs : int,
        the sampling frequency
    config : dict, optional,
        the configuration, with the following keys:
        - ``'wavelet_level'``: int,
            the level of the wavelet decomposition, default: 3
        - ``'wavelet_name'``: str,
            the name of the wavelet, default: "db7"

    Returns
    -------
    dwt_features : np.ndarray,
        the full DWT features, of shape (``'wavelet_level'``, nsamples)

    """
    cfg = ED(
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    siglen = len(signal)

    detail_coefs = pywt.wavedec(signal, cfg.wavelet_name, level=cfg.wavelet_level)[
        :0:-1
    ]
    dwt_features = np.zeros((cfg.wavelet_level, siglen), dtype=signal.dtype)
    for i, detail_coef in enumerate(detail_coefs):
        dwt_features[i] = _wkeep1(np.repeat(detail_coef, 2 ** (i + 1)), siglen)
    return dwt_features


def _wkeep1(x: np.ndarray, k: int, opt: Union[str, int] = "c") -> np.ndarray:
    """
    modified from the matlab function ``wkeep1``

    Parameters
    ----------
    x : np.ndarray,
        the input array
    k : int,
        the length of the output array
    opt : str or int, optional,
        specifies the position of the output array in the input array,
        if ``opt`` is an integer, then it is the first index of the output array,
        if ``opt`` is a string, then it can be one of the following:
        - ``"c"`` or ``"center"`` or ``"centre"``: the output array is centered in the input array
        - ``"l"`` or ``"left"``: the output array is left-aligned in the input array
        - ``"r"`` or ``"right"``: the output array is right-aligned in the input array

    Returns
    -------
    y : np.ndarray,
        the output array, of shape (k,),
        if ``k > len(x)``, then ``x`` is returned directly

    References
    ----------
    wkeep1.m of the matlab wavelet toolbox

    """
    x_len = len(x)
    if x_len <= k:
        return x
    if isinstance(opt, int):
        first = opt
    elif opt.lower() in ["c", "center", "centre"]:
        first = (x_len - k) // 2
    elif opt.lower() in ["l", "left"]:
        first = 0
    elif opt.lower() in ["r", "right"]:
        first = x_len - k
    else:
        raise ValueError(f"Unknown option: {opt}")
    assert 0 <= first <= x_len - k, f"Invalid first index: {first}"
    return x[first : first + k]
