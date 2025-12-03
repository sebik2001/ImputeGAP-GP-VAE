import time
from imputegap.wrapper.AlgoPython.GPVAE.runnerGPVAE import gpvae_recovery


def gp_vae(incomp_data, config_yaml_path, model_checkpoint_path=None, epoch=1, batch_size=64, beta=0.2, learning_rate=0.001, sigma=1.0, length_scale=7.0, kernel_scales=1, verbose=True, logs=True):
    """
    Perform imputation using the BRITS algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    config_yaml_path : str
        The path to the config .yaml file for the dataset
    model_checkpoint_path : str
        Specify the path to a trained model, use an already trained model. If None
        then a training will be done.
    epoch : int
        Number of epochs for training the model. Determines how many times the algorithm processes the entire dataset during training. If no training is needed it is ignored (default is 1)
    batch_size : int
        Size of the batches used during training. Larger batch sizes can speed up training but may require more memory (default is 64)
    beta: float
        Factor to weigh the KL term (similar to beta-VAE) (default is 0.2)
    learning_rate: float
        Learning rate for training (default: 0.001)
    sigma: float
        Sigma value for the GP prior (default is 1.0)
    length_scale: float
        Length scale value for the GP prior (default is 7.0)
    kernel_scales: int
        number of different length scales for the GP prior, length_scale/2^0, length_scale/2^1, ..., length_scale/2^i, with i = [0, kernel_scales - 1] (default is 1)
    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The GP-VAE algorithm is a machine learning-based approach for time series imputation, where missing values are recovered using a Variational Auto-Encoder (VAE) with a Gaussian process (GP) prior.

    Example
    -------
        >>> recov_data = gp_vae(...)
        >>> print(recov_data)

    References
    ----------
    Fortuin, V., Baranchuk, D., Rätsch, G. & Mandt, S. GP-VAE: Deep Probabilistic Multivariate Time Series Imputation. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 108: 1651–1661 (2020). https://proceedings.mlr.press/v108/fortuin20a/fortuin20a.pdf
    """
    start_time = time.time()  # Record start time

    # Imputation
    recov_data = gpvae_recovery(incomp_data,config_yaml_path, model_checkpoint_path=model_checkpoint_path, epoch=epoch, batch_size=batch_size, beta=beta, learning_rate=learning_rate, sigma=sigma, length_scale=length_scale, kernel_scales=kernel_scales, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation gpvae - Execution Time: {(end_time - start_time):.4f} seconds\n")


    return recov_data
