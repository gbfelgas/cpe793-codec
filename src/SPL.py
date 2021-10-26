import numpy as np
from scipy import signal

class SPL():

    def __init__(self, N_FFT: int, N_w: int, window_func: function):
        self.N_FFT = N_FFT
        self.N_w = N_w
        self.window = window_func(N_w)
        self.window_energy = np.sum(self.window ** 2) / N_w

    def spl_peaks(self, X: np.ndarray):
        """
        Calculates SPL for each signal peak considering energy of peak into 
        3 * self.N_FFT / (2 * self.window_energy) for each side of central sample.

        Args:
            X (np.ndarray): MDCT transform of x signal.

        Returns:
            SPL (np.ndarray): SPL equivalent of x signal, same size as X.
        """
        bw_factor = self.N_FFT / (2 * self.window_energy)
        norm = (self.window_energy * self.N_FFT / 8)
        
        X_db = 20 * np.log10(np.abs(X) + np.finfo(float).eps)

        peaks, _ = signal.find_peaks(X_db, height=0.1, prominence=25)
        SPL = np.zeros(peaks.shape)
        for i, p in enumerate(peaks):
            peak_energy = np.sum(np.abs(X[p - int(np.ceil(3 * bw_factor)):p + int(np.ceil(3 * bw_factor))]) ** 2) / norm
            SPL[i] = 96 + 10 * np.log10(peak_energy + np.finfo(float).eps)
        
        return SPL, peaks

    def spl_by_sample(self, X: np.ndarray):
        """
        Calculates SPL for each signal sample considering total peak energy on its central sample.

        Args:
            X (np.ndarray): MDCT transform of x signal.

        Returns:
            SPL (np.ndarray): SPL equivalent of x signal, same size as X.
        """
        norm = (self.window_energy * self.N_FFT / 8)

        peak_energy = np.abs(X) ** 2 / norm
        return 96 + 10 * np.log10(peak_energy + np.finfo(float).eps)