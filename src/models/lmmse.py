import numpy as np

class LMMSE:
    def __init__(
        self, 
        num_pilot_subcarriers: int,
        num_pilot_symbols: int,
        num_subcarriers: int,
        num_symbols: int):

        self.num_pilot_subcarriers = num_pilot_subcarriers
        self.num_pilot_symbols = num_pilot_symbols
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
    
    def __call__(self, hp_ls, sigma):
        sigma2_eye = np.eye(self.pilot_autocorr.shape[0]) * sigma
        pilot_autocorr_inv = np.linalg.inv(self.pilot_autocorr + sigma2_eye)
        h_true_hat = self.channel_pilot_crosscorr @ pilot_autocorr_inv @ hp_ls
        return h_true_hat

    def fit(self, dataloader):
        size_hp_ls = self.num_pilot_subcarriers * self.num_pilot_symbols
        size_h_true = self.num_subcarriers * self.num_symbols
        self.pilot_autocorr = np.zeros((size_hp_ls, size_hp_ls), dtype=np.complex128)
        self.channel_pilot_crosscorr = np.zeros((size_h_true, size_hp_ls), dtype=np.complex128)

        for batch in dataloader:
            hp_ls, h_true, _ = batch
            hp_ls = hp_ls.numpy()
            h_true = h_true.numpy()
            pilot_autocorr += self._compute_autocorrelation(hp_ls)
            channel_pilot_crosscorr += self._compute_cross_correlation(hp_ls, h_true)

        self.pilot_autocorr /= len(dataloader)
        self.channel_pilot_crosscorr /= len(dataloader)

    @staticmethod
    def _compute_autocorrelation(hp_ls):
        if hp_ls.ndim == 2:
            hp_ls = hp_ls.unsqueeze(0)  # add batch dimension
        assert hp_ls.ndim == 3

        batch_size = hp_ls.shape[0]

        hp_ls = hp_ls.reshape(batch_size, -1)  # row-major flatten

        hp_ls_autocorr = np.conj(hp_ls.T) @ hp_ls
        return hp_ls_autocorr / batch_size

    @staticmethod
    def _compute_cross_correlation(hp_ls, h_true):
        if hp_ls.ndim == 2:
            hp_ls = hp_ls.unsqueeze(0)  # add batch dimension
        
        if h_true.ndim == 2:
            h_true = h_true.unsqueeze(0)  # add batch dimension
        
        batch_size = hp_ls.shape[0]

        assert hp_ls.ndim == 3 and h_true.ndim == 3
        assert hp_ls.shape[0] == h_true.shape[0]

        hp_ls = hp_ls.reshape(batch_size, -1)  # row-major flatten
        h_true = h_true.reshape(batch_size, -1)  # row-major flatten

        hp_ls_h_true_cross_corr = np.conj(h_true.T) @ hp_ls
        
        return hp_ls_h_true_cross_corr/ batch_size
