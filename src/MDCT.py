import numpy as np

class MDCT():
    def __init__(self, N: int, window: np.ndarray):
        K = N // 2
        self.n_0 = ((N / 2) + 1) / 2

        A = np.zeros((K, N))
        for k in range(K):
            for n in range(N):
                A[k, n] = np.cos((2 * np.pi / N)*(n + self.n_0) * (k + 1 / 2))

        self.N = N
        self.window = window
        self.A_1 = A[:, :N//2]
        self.A_2 = A[:, N//2:]
        self.B_1 = (4 / N) * self.A_1.T
        self.B_2 = (4 / N) * self.A_2.T

        self.A = A
        self.B = (4 / N) * A.T

        if window.shape[0] != N:
            raise ValueError(f"Window is not the same size as N! Window size: {window.shape[0]}")

    def _preprocess(self, x: np.ndarray):
        L = len(x)
        self.frames = L // (self.N // 2) + 2
        r = L % (self.N // 2)
        self.extra_pad = (self.N // 2) - r
        return np.pad(x, (self.N // 2, (self.N // 2) + self.extra_pad))


    def mdct(self, x: np.ndarray):
        x_padded = self._preprocess(x)

        X = np.ndarray((self.A.shape[0], self.frames))
        for frame_idx in range(self.frames):
            frame = x_padded[frame_idx * (self.N // 2):frame_idx * (self.N // 2) + self.N]
            X[:, frame_idx] = self.A @ (frame * self.window)

        return X

    def imdct(self, X: np.ndarray):
        x_padded = np.zeros(self.frames * self.N // 2 + self.N // 2)
        for frame_idx in range(self.frames):
            x_padded[frame_idx * (self.N // 2):frame_idx * (self.N // 2) + self.N] += (self.B @ X[:, frame_idx]) * self.window
            
        return x_padded[(self.N // 2):- (self.N // 2) - self.extra_pad]
    
    def mdct_pre_twiddle(self):
        return np.exp(-2j * np.pi * np.arange(self.N) / (2 * self.N))

    def mdct_post_twiddle(self):
        return np.exp(-2j * np.pi * self.n_0 * (np.arange(self.N // 2) + 0.5) / self.N)


    def mdct_via_fft(self, x):
        x_padded = self._preprocess(x)

        X = np.ndarray((self.A.shape[0], self.frames))
        for frame_idx in range(self.frames):
            frame = x_padded[frame_idx * (self.N // 2):frame_idx * (self.N // 2) + self.N]
            X[:, frame_idx] = np.real(
                self.mdct_post_twiddle() * np.fft.fft(frame * self.window * self.mdct_pre_twiddle())[:(self.N // 2)]
            )
        
        return X 
    
    def imdct_pre_twiddle(self):
        return np.exp(2j * np.pi * self.n_0 * np.arange(self.N) / self.N)

    def imdct_post_twiddle(self):
        return np.exp(2j * np.pi * (np.arange(self.N) + self.n_0) / (2 * self.N))
    
    def imdct_via_ifft(self, X):
        x_padded = np.zeros((self.frames + 1) * (self.N // 2))
        for frame_idx in range(self.frames):
            x_padded[frame_idx * (self.N // 2):frame_idx * (self.N // 2) + self.N] += \
            np.real(np.fft.ifft(np.concatenate((X[:, frame_idx], -X[::-1, frame_idx])) * self.imdct_pre_twiddle()) * \
            self.imdct_post_twiddle()) * 2 * self.window
            
        return x_padded[(self.N // 2):- (self.N // 2) - self.extra_pad]