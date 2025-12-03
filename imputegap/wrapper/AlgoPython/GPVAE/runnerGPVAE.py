import os
import yaml
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from imputegap.wrapper.AlgoPython.GPVAE.models.models import BandedJointEncoder, GP_VAE, GaussianDecoder, BernoulliDecoder, ImagePreprocessor 


def train(model, incomp_data, m_mask, splits, nbr_features, seq_length, latent_dim, batch_size, epoch, learning_rate, gradient_clip, verbose=True):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if verbose:
        # pass a dummy input to both encoder and decoder in order to get the summaries
        dummy_encoder = tf.zeros([1, seq_length, nbr_features])
        dummy_decoder = tf.zeros([1, seq_length, latent_dim])
        model.encoder(dummy_encoder)
        model.decoder(dummy_decoder)
        print("Encoder: ", model.encoder.net.summary())
        print("Decoder: ", model.decoder.net.summary())
    
    if model.preprocessor is not None:
        # print("Preprocessor: ", model.preprocessor.net.summary())
        saver = tf.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net,
                                              decoder=model.decoder.net, preprocessor=model.preprocessor.net)
    else:
        saver = tf.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net, decoder=model.decoder.net)

    # TODO: set how to pass the imputegap assets
    outdir = './imputegap_assets/models/' + "exp_test"
    checkpoint_prefix = os.path.join(outdir, "ckpt")
    summary_writer = tf.summary.create_file_writer(outdir)
    summary_writer.set_as_default()

    x_train_miss = incomp_data[:splits[0]]
    m_train_miss = m_mask[:splits[0]]

    x_val_miss = incomp_data[:splits[0]]
    m_val_miss = incomp_data[:splits[0]]

    num_steps = epoch * len(x_train_miss) // batch_size

    print_interval = num_steps // epoch

    tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss))\
                                     .shuffle(len(x_train_miss)).batch(batch_size).repeat()
    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss)).batch(batch_size).repeat()
    tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)

    losses_train = []
    losses_val = []

    t0 = time.time()
    global_step = 0
    with summary_writer.as_default():
        for i, (x_seq, m_seq) in tqdm(enumerate(tf_x_train_miss.take(num_steps)), total=num_steps, desc="Training progress"):
            try:
                with tf.GradientTape() as tape:
                    # tape.watch(trainable_vars)
                    loss = model.compute_loss(x_seq, m_mask=m_seq)
                    losses_train.append(loss.numpy())
                
                trainable_vars = model.get_trainable_vars()
                grads = tape.gradient(loss, trainable_vars)
                grads = [np.nan_to_num(grad) for grad in grads]
                grads, global_norm = tf.clip_by_global_norm(grads, gradient_clip)
                optimizer.apply_gradients(zip(grads, trainable_vars))

                # Update progress bar postfix
                # tqdm.write(f"[Step {i}] Loss: {loss.numpy():.3f}, Grad norm: {global_norm:.2f}")

                # Print intermediate results
                if i % print_interval == 0:
                    print("================================================")
                    print("Learning rate: {} | Global gradient norm: {:.2f}".format(optimizer.learning_rate, global_norm))
                    print("Step {}) Time = {:2f}".format(i, time.time() - t0))
                    loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                    print("Train loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(loss, nll, kl))

                    saver.save(checkpoint_prefix)
                    tf.summary.scalar("loss_train", loss, step=global_step)
                    tf.summary.scalar("kl_train", kl, step=global_step)
                    tf.summary.scalar("nll_train", nll, step=global_step)
                    summary_writer.flush()

                    # Validation loss
                    x_val_batch, m_val_batch = tf_x_val_miss.get_next()
                    val_loss, val_nll, val_kl = model.compute_loss(x_val_batch, m_mask=m_val_batch, return_parts=True)
                    losses_val.append(val_loss.numpy())
                    print("Validation loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(val_loss, val_nll, val_kl))

                    tf.summary.scalar("loss_val", val_loss, step=global_step)
                    tf.summary.scalar("kl_val", val_kl, step=global_step)
                    tf.summary.scalar("nll_val", val_nll, step=global_step)
                    summary_writer.flush()

                    global_step += 1


                    # if FLAGS.data_type in ["hmnist", "sprites"]:
                    #     # Draw reconstructed images
                    #     x_hat = model.decode(model.encode(x_seq).sample()).mean()
                    #     tf.summary.image("input_train", tf.reshape(x_seq, [-1]+list(img_shape)), step=global_step)
                    #     tf.summary.image("reconstruction_train", tf.reshape(x_hat, [-1]+list(img_shape)), step=global_step)
                    # elif FLAGS.data_type == 'physionet':
                    #     # Eval MSE and AUROC on entire val set
                    #     x_val_miss_batches = np.array_split(x_val_miss, FLAGS.batch_size, axis=0)
                    #     x_val_full_batches = np.array_split(x_val_full, FLAGS.batch_size, axis=0)
                    #     m_val_artificial_batches = np.array_split(m_val_artificial, FLAGS.batch_size, axis=0)
                    #     get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_artificial_batches)

                    #     n_missings = m_val_artificial.sum()
                    #     mse_miss = np.sum([model.compute_mse(x, y=y, m_mask=m).numpy()
                    #                        for x, y, m in get_val_batches()]) / n_missings

                    #     x_val_imputed = np.vstack([model.decode(model.encode(x_batch).mean()).mean().numpy()
                    #                                for x_batch in x_val_miss_batches])
                    #     x_val_imputed[m_val_miss == 0] = x_val_miss[m_val_miss == 0]  # impute gt observed values

                    #     x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])
                    #     val_split = len(x_val_imputed) // 2
                    #     cls_model = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000)
                    #     cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
                    #     probs = cls_model.predict_proba(x_val_imputed[val_split:])[:, 1]
                    #     auroc = roc_auc_score(y_val[val_split:], probs)
                    #     print("MSE miss: {:.4f} | AUROC: {:.4f}".format(mse_miss, auroc))

                    #     # Update learning rate (used only for physionet with decay=0.5)
                    #     if i > 0 and i % (10*FLAGS.print_interval) == 0:
                    #         # optimizer._lr = max(0.5 * optimizer._lr, 0.1 * FLAGS.learning_rate)
                    #         new_lr = max(0.5 * float(optimizer.learning_rate.numpy()), 0.1 * FLAGS.learning_rate)
                    #         optimizer.learning_rate.assign(new_lr)
                    # t0 = time.time()
            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                break


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

def gpvae_recovery(incomp_data, config_yaml_path, model_checkpoint_path=None, epoch=None, batch_size=None, beta=None, learning_rate=None, sigma=None, length_scale=None, kernel_scales=None, verbose=True, seed=77):
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    # ----------------------- Load YAML config -----------------------------
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config YAML path '{config_yaml_path}' does not exist.")
    
    with open(config_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # -------------------- Model hyperparameters from YAML ----------------
    splits = cfg["splits"]
    latent_dim = cfg["latent_dim"]
    nbr_features = cfg["nbr_features"]
    seq_length = cfg["seq_length"]
    encoder_sizes = cfg["encoder_sizes"]
    cov_activation = cfg['cov_activation']
    decoder_sizes = cfg["decoder_sizes"]
    window_size = cfg.get("window_size", 24)
    kernel = cfg.get("kernel", "cauchy")
    image_preprocessor = cfg.get("image_preprocessor", False)
    if image_preprocessor:
        image_preprocessor = ImagePreprocessor(cfg["image_shape"], cfg['cnn_sizes'], cfg['cnn_kernel_size'])
    else:
        image_preprocessor = None
    M = cfg.get("M", 1)
    K = cfg.get("K", 1)

    # Select decoder class based on YAML or default
    decoder_type = cfg.get("encoder_type", "GaussianDecoder")
    if decoder_type == "GaussianDecoder":
        decoder = GaussianDecoder
    elif decoder_type == "BernoulliDecoder":
        decoder = BernoulliDecoder
    else:
        raise ValueError(f"Unknown decoder class: {decoder_type}")
    
    gradient_clip = cfg.get("gradient_clip", 10000.0)
    
    # use the one from the config if not passed as arguments
    epoch = cfg["epoch"] if epoch is None else epoch
    batch_size = cfg["batch_size"] if batch_size is None else batch_size
    beta = cfg["beta"] if beta is None else beta
    learning_rate = cfg["learning_rate"] if learning_rate is None else learning_rate
    sigma = cfg["sigma"] if sigma is None else sigma
    length_scale = cfg["length_scale"] if length_scale is None else length_scale
    kernel_scales = cfg["kernel_scales"] if kernel_scales is None else kernel_scales
    
    #---------------------- Reshape the data to (B,V,T) ------------------------
    recov = recov.transpose().reshape(-1, seq_length, nbr_features)  
    incomp_data = incomp_data.transpose().reshape(-1, seq_length, nbr_features)
    m_mask = m_mask.transpose().reshape(-1, seq_length, nbr_features)

    # cast to float32 (net compatibility)
    incomp_data = incomp_data.astype(np.float32)
    # m_mask = m_mask.astype(np.float32)

    #---------------------- Build the model ---------------------------------
    model = GP_VAE(
        latent_dim=latent_dim, 
        data_dim=nbr_features, 
        time_length=seq_length,
        encoder_sizes=encoder_sizes,
        encoder=BandedJointEncoder,
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
        cov_activation=cov_activation
    )

    #---------------- Reload the model ------------------------------
    if model_checkpoint_path and os.path.exists(model_checkpoint_path):
        if image_preprocessor:
            checkpoint = tf.train.Checkpoint(
                encoder=model.encoder.net,
                decoder=model.decoder.net,
                preprocessor=model.preprocessor.net
            )
        else:
            checkpoint = tf.train.Checkpoint(
                encoder=model.encoder.net,
                decoder=model.decoder.net,
            )

        # restore the latest checkpoint
        latest = tf.train.latest_checkpoint(model_checkpoint_path)
        checkpoint.restore(latest).expect_partial()
        print("Checkpoint successfully restored.")
        
    # --------------Train the model -------------------------------
    else:
        # TODO train the model
        print("Start model training...")
        train(model, incomp_data, m_mask, splits, nbr_features, seq_length, latent_dim, batch_size, epoch, learning_rate, gradient_clip, verbose=verbose)
        print("Finished training")
    

    # ---------------- Impute data --------------------------------------
    # incomp_data contains NaN values, set NaN to 0.0
    incomp_data_model = np.nan_to_num(incomp_data, nan=0.0)

    # pass the incomp data to the encoder
    encoder_output = model.encode(incomp_data_model)

    # pass enconder's output most probable latent trajectory to the decoder
    decoder_output = model.decode(encoder_output.mean())

    # most probable data space trajectory
    recovery = decoder_output.mean().numpy()

    recov[m_mask] = recovery[m_mask]

    # reshape (B, V, T) -> (T, B*V)
    recov = recov.transpose(2,0,1).reshape(nbr_features, -1)
    return recov
