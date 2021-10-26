import numpy as np


def uniform_allocation(total_bits, K):
    initial_allocation = (total_bits) // K * np.ones((K))
    remaining_bits = total_bits % K
    initial_allocation[np.random.permutation(K)[:remaining_bits]] += 1
    
    return initial_allocation

def normal_allocation(
    K_p: int,
    total_bits: int,
    N_b: np.ndarray, 
    value_b: np.ndarray,
    perceptual: bool = True
):
    if perceptual:
        avg = np.sum(value_b * np.array(N_b)) / K_p
        buffer = np.array((total_bits / K_p) + (np.log(10) / (20 * np.log(2))) * (value_b - avg))
            
        return np.floor(buffer)

    avg = np.sum(0.5 * np.array(N_b) * np.log2(value_b ** 2)) / K_p
    buffer = np.array((total_bits / K_p) + 0.5 * np.log2(value_b ** 2) - avg)
        
    return np.floor(buffer)

def greedy_allocation(
    K_p: int,
    total_bits: int,
    N_b: np.ndarray, 
    value_b: np.ndarray,
    perceptual: bool = True
):
    if perceptual:
        avg = np.sum(np.array(N_b) * value_b) / K_p
        buffer = np.array((total_bits / K_p) + (np.log(10) / (20 * np.log(2))) * (value_b - avg))
        
        positives = np.ones_like(buffer)
        while np.any(buffer < 0):
            positives[np.nonzero(buffer < 0)[0]] = 0
            K_p = np.sum(N_b * positives)
            
            avg = np.sum(np.array(N_b) * value_b * positives)/K_p
            buffer = np.array((total_bits / K_p) + (np.log(10) / (20 * np.log(2))) * (value_b - avg)) * positives
            
        return np.floor(buffer)

    avg = np.sum(0.5 * np.array(N_b) * np.log2(value_b ** 2)) / K_p
    buffer = np.array((total_bits / K_p) + 0.5 * np.log2(value_b ** 2) - avg)
    
    positives = np.ones_like(buffer)
    while np.any(buffer < 0):
        positives[np.nonzero(buffer < 0)[0]] = 0
        K_p = np.sum(N_b * positives)
        
        avg = np.sum(0.5 * np.array(N_b) * np.log2(value_b ** 2) * positives) / K_p
        buffer = np.array((total_bits / K_p) + 0.5 * np.log2(value_b ** 2) - avg) * positives
        
    return np.floor(buffer)

def waterfilling(
    value_b: np.ndarray,
    N_b: np.ndarray,
    total_bits: int,
    perceptual: bool = True
):
    if perceptual:
        threshold = np.max(value_b)
        t_inf = np.min(value_b)

    else:
        threshold = np.log2(np.max(value_b))
        t_inf = np.log2(np.min(value_b))
    
    sub_band_bits = np.zeros_like(value_b)
    bits_allocated = 0
    sorted_idxs = np.argsort(value_b)[::-1]
    remaining_bits = total_bits
    
    while bits_allocated < total_bits:
        for i in sorted_idxs:
            if perceptual:
                limit = value_b[i]
            else:
                limit = np.log2(value_b[i])

            if limit >= threshold and bits_allocated < total_bits and remaining_bits >= N_b[i]:
                sub_band_bits[i] += N_b[i]
                bits_allocated += N_b[i]
                remaining_bits = total_bits - bits_allocated
                
            if remaining_bits < N_b[i]:
                return sub_band_bits
            
        if threshold > t_inf:
            if perceptual:
                threshold -= 6
            else:
                threshold -= 1
        else:
            threshold = t_inf
            
    return sub_band_bits