import numpy as np

def bark(f: int):
    return 13 * np.arctan(.76 * f / 1000) + 3.5 * np.arctan((f / 7500) ** 2)

def get_subband_index(f: int):
    return 1 + int(bark(f))

def bin_to_subband_index(k: int, fs: int = 48000, N: int = 2048):
    return 1 + int(bark(k * fs / N))

def R_k(data_rate: int, fs: int):
    return data_rate / fs

def block_bits_per_channel(R_k: float, N: int):
    return R_k * N / 2

def sine_window(N):
    M = N//2
    w = np.zeros((N,1))

    for n in range(N-M):
        w[n] = np.sin((np.pi/2)*(n+1/2)/(N-M))
        
    for n in range(N-M,M):
        w[n] = 1

    for n in range(M,N):
        w[n] = np.sin((np.pi/2)*(N - n - 1/2)/(N-M))

    return w.reshape(-1)