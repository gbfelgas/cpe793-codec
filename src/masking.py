import numpy as np
from src.utils import bark

def delta_tone_masking_noise(z, delta):
    return delta + z

def masking_model(bark, SPL, peaks, delta=15):
    sf_peaks = np.zeros((len(bark), len(bark)))
    
    teta = np.zeros_like(bark)
    for i in peaks:
        dz = bark - bark[i]
        teta[dz > 0] = 1
        
        # spreading function
        sf = ((-27 + 0.37 * np.max((SPL[i] - 40), 0) * teta) * np.abs(dz))
        
        # masking model
        sf_peaks[i,:] = np.maximum(SPL[i] + sf - delta_tone_masking_noise(bark[i], delta), 0)
     
    return sf_peaks

def hearing_threshold(mdct_freqs):
    """
    Threshold in quiet
    """
    mdct_freqs[0] = mdct_freqs[1]
    threshold = (3.64 * (mdct_freqs / 1000) ** (-.8)) - 6.5 * np.exp(-.6 * ((mdct_freqs / 1000) - 3.3) ** 2) + 10 ** (-3) * (mdct_freqs / 1000) ** 4
    threshold_barks = bark(mdct_freqs)
    return (threshold_barks, threshold)

def inf_norm_mask(mask_1, mask_2):
    return np.maximum(mask_1, mask_2)