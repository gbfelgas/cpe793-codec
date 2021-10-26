import numpy as np
from src import MDCT
from src.utils import sine_window, bin_to_subband_index, block_bits_per_channel, bark, R_k
from src.allocations import uniform_allocation, normal_allocation, greedy_allocation, waterfilling
from src.masking import spl, spl_by_sample, spl_peaks, masking_model, hearing_threshold, inf_norm_mask

def count_zeros(number: int, R: int):
    """
    Shifts number by one until leftmost bit does not become 1.
    
    Args:
        number (int): Number to count leading zeros. 
        R (int): Total number of bits, defined by (2 ** R_s) - 1 + R_m
    """
    
    res = 0
    if number == 0:
        return R
    while ((number & (1 << (R - 1))) == 0):
        number = (number << 1)
        res += 1
 
    return res

def code_uniform_midtread(number, R):
    if number >= 0:
        signal = 1
    else:
        signal = -1
    if abs(number) >= 1:
        output = 2 ** (R - 1) - 1
    else:
        output = int(((2 ** R - 1) * abs(number) + 1) / 2)
    
    return signal * output

def decode_uniform_midtread(code,R):
    signal = 2 * int(code & 2 ** (R - 1) == 0) - 1 
    number_abs = 2 * abs(code) / (2 ** R - 1)
    return signal * number_abs

def get_scale(x):
    return 2 ** (-count_zeros(code_uniform_midtread(x , 6), 6) + 1)

class BibsCodec():

    def __init__(
        self,
        N: int,
        fs: int,
        n_bits_mantissa_b: int,
        n_bits_scale: int = 4
    ):
        self.N = N
        self.fs = fs
        self.n_bits_mantissa_b = n_bits_mantissa_b
        self.n_bits_scale = n_bits_scale

    def block_fp_quantizer(samples_b, n_bits_mantissa_b, n_bits_scale: int = 4):
        N_b = samples_b.shape[0]
        R_m = int(n_bits_mantissa_b // N_b)
        r = int(n_bits_mantissa_b % N_b)
        
        if r == 0:
            R = int(2 ** n_bits_scale - 1 + R_m)
            scales = np.zeros_like(samples_b)
            for i, v in enumerate(samples_b):
                leading_zeros = count_zeros(code_uniform_midtread(np.abs(v), R), R) - 1
                if leading_zeros < 2 ** n_bits_scale - 1:
                    scales[i] = 2 ** n_bits_scale - 1 - leading_zeros
            max_scale = scales.max()
            quant_samples = np.zeros_like(samples_b)
            
            for i, v in enumerate(samples_b):
                quant_number = code_uniform_midtread(np.abs(v), R)
                quant_samples[i] = quant_number >> int(max_scale) # right bit shift
                s = np.abs(np.sign(v) - 1) // 2
                quant_samples[i] += int(s) << (R_m - 1) # left bit shift => primeiro bit da esquerda da mantissa
        else:
            mantissas = np.full_like(samples_b, R_m) # Primeiro, todas as amostras recebem as mantissas merecidas
            mantissas[:r] += 1 # Depois, as amostras que ganharam mais bits para mantissa as recebem
            scales = np.zeros_like(samples_b) # escalas das amostras de uma unica sub-banda
            for i, v in enumerate(samples_b):
                R = 2 ** n_bits_scale - 1 + int(mantissas[i])
                leading_zeros = count_zeros(code_uniform_midtread(np.abs(v), R), R) - 1
                if leading_zeros < 2 ** n_bits_scale - 1:
                    scales[i] = 2 ** n_bits_scale - 1 - leading_zeros
            max_scale = scales.max()
            quant_samples = np.zeros_like(samples_b)
            
            for i, v in enumerate(samples_b):
                if (mantissas[i] != 0):
                    R = 2 ** n_bits_scale - 1 + int(mantissas[i])
                    quant_number = code_uniform_midtread(np.abs(v), R)
                    quant_samples[i] = quant_number >> int(max_scale) # right bit shift
                    s = np.abs(np.sign(v) - 1) // 2
                    quant_samples[i] += int(s) << (int(mantissas[i]) - 1) # left bit shift => primeiro bit da esquerda da mantissa
                else:
                    quant_samples[i] = 0
        return quant_samples, max_scale

    def block_fp_dequantizer(coded_samples_b,  block_scale, n_bits_mantissa_b, n_bits_scale: int = 4):
        N_b = coded_samples_b.shape[0]
        R_m = int(n_bits_mantissa_b // N_b)
        r = int(n_bits_mantissa_b % N_b)
        decoded_samples = np.zeros_like(coded_samples_b, dtype=float)
        if r == 0:
            R = 2 ** n_bits_scale - 1 + R_m
            for i, v in enumerate(coded_samples_b):
                sign_bit = int(int(v)&(2**(R_m-1))!=0)
                s = (-2)*sign_bit + 1
                if block_scale != 0:
                    v_no_s_bit = int(v) & (2**(R_m) - 1 - (1 << R_m - 1)) 
                    decoded_samples[i] = s*decode_uniform_midtread((int(v_no_s_bit) << int(block_scale)) + 0*(1 << int(block_scale) - 1), R)
                else:
                    v_no_s_bit = int(v) & (2**(R_m) - 1 - (1 << R_m - 1))
                    decoded_samples[i] = s*decode_uniform_midtread(v_no_s_bit, R)
        else:
            mantissas = np.full_like(coded_samples_b, R_m) # Primeiro, todas as amostras recebem as mantissas merecidas
            mantissas[:r] += 1 # Depois, as amostras que ganharam mais bits para mantissa as recebem
            for i, v in enumerate(coded_samples_b):
                if mantissas[i] != 0:
                    R = 2 ** n_bits_scale - 1 + int(mantissas[i])
                    sign_bit = int(int(v) & int(2**(int(mantissas[i])-1))!=0)
                    s = (-2)*sign_bit + 1
                    if block_scale != 0:
                        v_no_s_bit = int(v) & (2**(int(mantissas[i])) - 1 - (1 << int(mantissas[i]) - 1)) 
                        decoded_samples[i] = s*decode_uniform_midtread((int(v_no_s_bit) << int(block_scale)) + 0*(1 << int(block_scale) - 1), R)
                    else:
                        v_no_s_bit = int(v) & (2**(int(mantissas[i])) - 1 - (1 << int(mantissas[i]) - 1))
                        decoded_samples[i] = s*decode_uniform_midtread(v_no_s_bit, R)
                else:
                    decoded_samples[i] = 0
            
        return decoded_samples

    def code_and_decode_vector(
        input_vector: np.ndarray,
        N: int,
        fs: int,
        n_scale_bits: int,
        I: int,
        window: np.ndarray = None,
        allocation_scheme: str = 'uniform',
        allocation_strategy: str = 'waterfilling',
        scale_norm: bool = True
    ):
        if window == None:
            mdct_transformer = MDCT(N, sine_window(N))
        else:
            assert (N == window.shape[0])
            mdct_transformer = MDCT(N, window)
        
        K = 25 # Numero de subbandas
        input_mdct = mdct_transformer.mdct(input_vector)
        Frames = input_mdct.shape[1]
        quant_mdct = np.zeros_like(input_mdct)
        k_map = {k:0 for k in range(1, K+1)}
        for freq_bin in range(N//2): 
            k = bin_to_subband_index(freq_bin, fs, N)
            k_map[k] = k_map[k] + 1
            
        N_b = np.array(list(k_map.values()))
        Kp = N_b.sum()
        total_bits = int(block_bits_per_channel(R_k(I, fs), N) - K*n_scale_bits)
        max_value = np.max(input_mdct)
        for frame in range(Frames):
            
            current_frame = input_mdct[:, frame]
            # Calculo dos indices de frequência
            mdct_bins = np.arange(N // 2)
            mdct_freqs = (mdct_bins * fs / N)
            bark_freqs = bark(mdct_freqs)
            
            ### Alocação de bits:
            if allocation_scheme == 'uniform':
                allocation = uniform_allocation(total_bits, K)
                
            elif allocation_scheme == 'optimal_error':
                
                if scale_norm == False:
                    ### Transformar esse bloco em uma funcao, talvez?############
                    df = pd.DataFrame({'bin': range(N//2), 'mdct': current_frame})
                    df["subband"] = df["bin"].apply(bin_to_subband_index, args=(fs, N))
                    df.drop(columns = ["bin"], inplace = True)
                    df['scale_bits'] = df['mdct'].apply(lambda x: np.ceil(np.log2(np.abs(x) + 0.0001)))
                    x_max_b = df.groupby(by=["subband"]).max()["scale_bits"].to_numpy()
                    ##############################################################
                else:
                    df = pd.DataFrame({'bin': range(N//2), 'mdct': current_frame / max_value})
                    df["subband"] = df["bin"].apply(bin_to_subband_index, args=(fs, N))
                    df.drop(columns = ["bin"], inplace = True)
                    df['scale_bits'] = df['mdct'].apply(lambda x: max_value * get_scale(x))
                    x_max_b = df.groupby(by=["subband"]).max()["scale_bits"].to_numpy()

                
                if allocation_strategy == 'normal':
                    allocation = normal_allocation(Kp, total_bits, N_b, x_max_b, perceptual=False)
                    allocation = allocation * N_b
                    
                elif allocation_strategy == 'waterfilling':
                    allocation = waterfilling(x_max_b, N_b, total_bits, perceptual=False)

                elif allocation_strategy == 'greedy':
                    allocation = greedy_allocation(Kp, total_bits, N_b, x_max_b, perceptual=False)
                    allocation = allocation * N_b
                else:
                    raise ValueError("No Such Strategy as " + allocation_strategy)

                
            elif allocation_scheme == 'perceptual':
                
                if scale_norm == False:
                    SPL = spl_by_sample(current_frame, N, N)
                    SPL_peaks, peaks = spl_peaks(current_frame, N, N)
                    mask = np.max(masking_model(bark_freqs, SPL, peaks), axis=0)
                    t_bark, thres = hearing_threshold(mdct_freqs)
                    complete_mask = inf_norm_mask(thres, mask)
                    SMR = SPL - complete_mask
                else:
                    df = pd.DataFrame({'bin': range(N//2), 'mdct': current_frame / max_value})
                    df["subband"] = df["bin"].apply(bin_to_subband_index, args=(fs, N))
                    df.drop(columns = ["bin"], inplace = True)
                    df['scale_bits'] = df['mdct'].apply(lambda x: max_value * get_scale(x))
                    x_max_b = df.groupby(by=["subband"]).max()["scale_bits"].to_numpy()
                    
                    SPL = spl(x_max_b, N, N)
                    SPL_peaks, peaks = spl_peaks(current_frame, N, N)
                    mask = np.max(masking_model(bark_freqs, SPL, peaks), axis=0)
                    t_bark, thres = hearing_threshold(mdct_freqs)
                    complete_mask = inf_norm_mask(thres, mask)
                    SMR = SPL - complete_mask
                
                ### Transformar esse bloco em uma funcao, talvez?#########
                df = pd.DataFrame({'bin': range(N//2), 'smr': SMR})
                df["subband"] = df["bin"].apply(bin_to_subband_index, args=(fs, N))
                df.drop(columns = ["bin"], inplace = True)
                smr_b = df.groupby(by=["subband"]).max()["smr"].to_numpy()
                ##########################################################
                
                if allocation_strategy == 'normal':
                    allocation = bit_allocation_perceptual(Kp, total_bits, N_b, smr_b)
                    allocation = allocation * N_b
                elif allocation_strategy == 'waterfilling':
                    allocation = waterfilling_perceptual(smr_b, N_b, total_bits)
                
                elif allocation_strategy == 'greedy':
                    allocation = bit_allocation_perceptual_greedy(Kp, total_bits, N_b, smr_b)
                    allocation = allocation * N_b
                
                else:
                    raise ValueError("No Such Strategy as " + allocation_strategy)
            else:
                raise ValueError("No Such Scheme as " + allocation_scheme)
            
            ## Quantização e Dequantização
            normalization_constant = max_value
            current_frame = current_frame/normalization_constant
            for k in range(K):
                mantissa_bits = allocation[k] if not np.isnan(allocation[k]) else 0 # Mantissa bits for that subband
                if mantissa_bits != 0:
                    coded_samples, band_scale = block_fp_quantizer(current_frame[np.floor(bark_freqs) == k], mantissa_bits, n_scale_bits)
                    decoded_samples = block_fp_dequantizer(coded_samples, band_scale, mantissa_bits, n_scale_bits)
                    quant_mdct[np.floor(bark_freqs) == k, frame] = decoded_samples
                else:
                    quant_mdct[np.floor(bark_freqs) == k, frame] = 0
            
            quant_mdct[:, frame] *= normalization_constant
            
        output_vector = mdct_transformer.imdct(quant_mdct)
        return output_vector
            
    