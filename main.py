import numpy as np
from src import MDCT
from src.utils import sine_window


def main(args):
    N = 2048
    Fs = 48000

    n = np.arange(N)
    T = N / Fs
    t = np.linspace(0, T, N)

    A0 = 0.6
    A1 = 0.55
    A2 = 0.55
    A3 = 0.15
    A4 = 0.1
    A5 = 0.05

    f0 = 440
    f1 = 554
    f2 = 660
    f3 = 880
    f4 = 4400
    f5 = 8800


    example_signal = (A0 * np.cos(2 * np.pi * n * f0/Fs)) \
    + (A1 * np.cos(2 * np.pi * n * f1/Fs)) \
    + (A2 * np.cos(2 * np.pi * n * f2/Fs)) \
    + (A3 * np.cos(2 * np.pi * n * f3/Fs)) \
    + (A4 * np.cos(2 * np.pi * n * f4/Fs)) \
    + (A5 * np.cos(2 * np.pi * n * f5/Fs))


    example_signal = example_signal / 2
    my_mdct = MDCT(N, sine_window(N))

    ex_mdct = my_mdct.mdct(example_signal)[:, 1]
    ex_mdct_max = np.max(np.abs(ex_mdct))

    mdct_bins = np.arange(ex_mdct.shape[0])
    mdct_freqs = (mdct_bins * Fs / N)

if __name__ == '__main__':
    main(args)