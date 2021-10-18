import numpy as np
import matplotlib.mlab as mlab

# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
DEFAULT_FS = 44100

# Size of the FFT window,
# The number of data points used in each block for the FFT.
# affects frequency granularity
DEFAULT_WINDOW_SIZE = 4096

# Ratio by which each sequential window overlaps the last and the
# next window. Higher overlap will allow a higher granularity of offset
# matching, but potentially more fingerprints.
DEFAULT_OVERLAP_RATIO = 0.5


def get_spectrogram(channel_samples, Fs=DEFAULT_FS, wsize=DEFAULT_WINDOW_SIZE, wratio=DEFAULT_OVERLAP_RATIO):

    # SFFT the signal and extract frequency components
    spectrogram = mlab.specgram(
            channel_samples,
            NFFT=wsize,
            Fs=Fs,
            window=mlab.window_hanning,
            noverlap=int(wsize * wratio))[0]

    # Apply log transform since specgram function returns linear array. 0s are excluded to avoid np warning
    spectrogram = 10 * np.log10(spectrogram, out=np.zeros_like(spectrogram),
                        where=(spectrogram != 0))
    return spectrogram
