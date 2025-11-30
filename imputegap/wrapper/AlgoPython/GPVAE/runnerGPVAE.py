import os

import numpy as np
import pandas as pd
import tensorflow as tf
from imputegap.wrapper.AlgoPython.GPVAE.models.models import BandedJointEncoder, GaussianDecoder, GP_VAE



# def train(model, input, batch_size, epochs, num_workers=0, verbose=True):
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     data_iter = data_loader.get_loader(input, batch_size=batch_size, num_workers=num_workers)
#     for epoch in range(0, epochs):
#         model.train()
#         run_loss = 0.0

#         for idx, data in enumerate(data_iter):  # 4
#             data = utilsX.to_var(data)
#             ret = model.run_on_batch(data, optimizer)
#             run_loss += ret['loss'].data

#             forward = data["forward"]
#             values = forward['values']

#             if verbose:
#                 print('\r Progress epoch {}, {:.2f}%, batch {} [{}], average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), idx, values.shape, run_loss / (idx + 1.0)))

#     return (model, data_iter)


# def evaluate(model, val_iter):
#     model.eval()
#     imputations = []

#     for idx, data in enumerate(val_iter):
#         data = utilsX.to_var(data)
#         ret = model.run_on_batch(data, None)
#         imputation = ret['imputations'].data.cpu().numpy()
#         imputations += imputation.tolist()

#     imputations = np.asarray(imputations)
#     return imputations

##################################################################
##################################################################
# parameters necessary that can be set on train.py
# Script to train the proposed GP-VAE model.


# flags:

# train.py:
#   --K: Number of importance sampling weights
#     (default: '1')
#     (an integer)
#   --M: Number of samples for ELBO estimation
#     (default: '1')
#     (an integer)
#   --[no]banded_covar: Use a banded covariance matrix instead of a diagonal one for the output of the inference network: Ignored if model_type is not gp-vae
#     (default: 'false')
#   --basedir: Directory where the models should be stored
#     (default: 'models')
#   --batch_size: Batch size for training
#     (default: '64')
#     (an integer)
#   --cnn_kernel_size: Kernel size for the CNN preprocessor
#     (default: '3')
#     (an integer)
#   --cnn_sizes: Number of filters for the layers of the CNN preprocessor
#     (default: '256')
#     (a comma separated list)
#   --data_dir: Directory from where the data should be read in
#     (default: '')
#   --data_type: <hmnist|physionet|sprites>: Type of data to be trained on
#     (default: 'hmnist')
#   --exp_name: Name of the experiment
#     (default: 'debug')
#   --gradient_clip: Maximum global gradient norm for the gradient clipping during training
#     (default: '10000.0')
#     (a number)
#   --kernel: <rbf|diffusion|matern|cauchy>: Kernel to be used for the GP prior: Ignored if model_type is not (m)gp-vae
#     (default: 'cauchy')
#   --kernel_scales: Number of different length scales sigma for the GP prior: Ignored if model_type is not gp-vae
#     (default: '1')
#     (an integer)
#   --learning_rate: Learning rate for training
#     (default: '0.001')
#     (a number)
#   --model_type: <vae|hi-vae|gp-vae>: Type of model to be trained
#     (default: 'gp-vae')
#   --num_steps: Number of training steps: If non-zero it overwrites num_epochs
#     (default: '0')
#     (an integer)
#   --print_interval: Interval for printing the loss and saving the model during training
#     (default: '0')
#     (an integer)
#   --seed: Seed for the random number generator
#     (default: '1337')
#     (an integer)
#   --[no]testing: Use the actual test set for testing
#     (default: 'false')
def gpvae_recovery(incomp_data, model_checkpoint_path=None):
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    if model_checkpoint_path:
        print('Path to a model checkpoint was given.')

        if not os.path.exists(model_checkpoint_path):
            print('The path to the model checkpoint is invalid')
            return
        
        # read results.tsv to infer model hyperparams used during training
        results = pd.read_csv(os.path.join(model_checkpoint_path, "results.tsv"), delimiter="\t")

        # build the model
        encoder = BandedJointEncoder
        decoder = GaussianDecoder
        image_preprocessor = None

        data_type = results.data[0]
        kernel = results.kernel[0]
        beta = results.beta[0]
        window_size = int(results.window_size[0])
        kernel_scales = results.kernel_scales[0]
        sigma = results.sigma[0]
        length_scale = results.length_scale[0]
        encoder_sizes = [results.encoder_width[0]]*results.encoder_depth[0]
        decoder_sizes = [results.decoder_width[0]]*results.decoder_depth[0]
        
        #################################################################
        # currently hardcoded for Physionet exp (TODO solve this issue)
        latent_dim = 35
        data_dim = 35
        time_length = 48
        M = 1
        K = 1
        #################################################################

        # build the model
        model = GP_VAE(
            latent_dim=latent_dim, 
            data_dim=data_dim, 
            time_length=time_length,
            encoder_sizes=encoder_sizes,
            encoder=encoder,
            decoder_sizes=decoder_sizes,
            decoder=decoder,
            kernel=kernel, 
            sigma=sigma,
            length_scale=length_scale, 
            kernel_scales = kernel_scales,
            image_preprocessor=image_preprocessor, 
            window_size=window_size,
            beta=beta, 
            M=M,
            K=K,
            data_type=data_type
        )

        # reload the checkpoint weights into the model
        if image_preprocessor is None:
            checkpoint = tf.train.Checkpoint(
                encoder=model.encoder.net,
                decoder=model.decoder.net,
            )
        else:
            checkpoint = tf.train.Checkpoint(
                encoder=model.encoder.net,
                decoder=model.decoder.net,
                preprocessor=model.preprocessor.net
            )

        # restore the latest checkpoint
        latest = tf.train.latest_checkpoint(model_checkpoint_path)
        checkpoint.restore(latest).expect_partial()
        print("Checkpoint successfully restored.")

        # incomp_data contains NaN values, set NaN to 0.0
        incomp_data_model = np.nan_to_num(incomp_data, nan=0.0)

        # incomp_data (T, V), T: time series, V: values, model expects
        # (B, V, T), B: batches, V: values, T: time series
        incomp_data_model = [incomp_data_model.transpose()]

        # pass the incomp data to the encoder
        encoder_output = model.encode(incomp_data_model)

        # pass enconder's output most probable latent trajectory to the decoder
        decoder_output = model.decode(encoder_output.mean())

        # most probable data space trajectory, (B,V,T) -> (T,V)
        recovery = decoder_output.mean().numpy()[0].transpose()

        recov[m_mask] = recovery[m_mask]

    return recov
