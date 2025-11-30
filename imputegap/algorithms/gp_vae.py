from imputegap.wrapper.AlgoPython.GPVAE.runnerGPVAE import gpvae_recovery


def gp_vae(incomp_data, model_checkpoint_path=None):
    """
    Perform imputation using the GP-VAE algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).
    params : dict, optional
        Optional parameters for the algorithm. If None, the minimum value from the contamination is used (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the mean value from the time series.

    Notes
    -----
    This function finds the non-NaN value in the time series and replaces all NaN values with this mean value.
    It is a simple imputation technique for filling missing data points in a dataset.

    Example
    -------
        >>> incomp_data = np.array([[5, 2, np.nan], [3, np.nan, 6]])
        >>> recov_data = mean_impute(incomp_data)
        >>> print(recov_data)
        array([[5., 2., 4.],
               [3., 4., 6.]])

    """

    # Imputation
    recov_data = gpvae_recovery(incomp_data=incomp_data, model_checkpoint_path=model_checkpoint_path)

    return recov_data
