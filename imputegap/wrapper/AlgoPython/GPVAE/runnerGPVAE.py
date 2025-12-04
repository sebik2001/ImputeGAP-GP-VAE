import os
import yaml
import time
import copy
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from imputegap.wrapper.AlgoPython.GPVAE.models.models import BandedJointEncoder, GP_VAE, GaussianDecoder, BernoulliDecoder, ImagePreprocessor 


def train(model, incomp_data, m_mask, splits, nbr_features, seq_length, latent_dim, batch_size, epoch, scheduler_cfg, learning_rate, gradient_clip, outdir, verbose=True):    

    if verbose:
        # pass a dummy input to both encoder and decoder in order to get the summaries
        dummy_encoder = tf.zeros([1, seq_length, nbr_features])
        dummy_decoder = tf.zeros([1, seq_length, latent_dim])
        model.encoder(dummy_encoder)
        model.decoder(dummy_decoder)
        print("Encoder: ", model.encoder.net.summary())
        print("Decoder: ", model.decoder.net.summary())
    
    checkpoint_prefix = os.path.join(outdir, "ckpt")
    summary_writer = tf.summary.create_file_writer(outdir)
    summary_writer.set_as_default()

    x_train_miss = incomp_data[:splits[0]]
    m_train_miss = m_mask[:splits[0]]

    x_val_miss = incomp_data[splits[0]:splits[0]+splits[1]]
    m_val_miss = incomp_data[splits[0]:splits[0]+splits[1]]

    num_steps = epoch * len(x_train_miss) // batch_size

    print_interval = num_steps // epoch

    decay_steps = print_interval*10

    if scheduler_cfg["type"] == "ExponentialDecay":
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=scheduler_cfg.get("decay_rate", 0.5),
            staircase=scheduler_cfg.get("staircase", True)
        )
    elif scheduler_cfg["type"] == "CosineDecay":
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=scheduler_cfg.get("alpha", 0.0)
        )
    else:
        lr_schedule = learning_rate  # constant learning rate
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    if model.preprocessor is not None:
        # print("Preprocessor: ", model.preprocessor.net.summary())
        saver = tf.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net,
                                              decoder=model.decoder.net, preprocessor=model.preprocessor.net)
    else:
        saver = tf.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net, decoder=model.decoder.net)
    
    tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss))\
                                     .shuffle(len(x_train_miss)).batch(batch_size).repeat()
    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss)).batch(batch_size)

    losses_train = []
    kl_losses_train = []
    nll_losses_train = []
    losses_val = []
    kl_losses_val = []
    nll_losses_val = []

    t0 = time.time()
    global_step = 0
    with summary_writer.as_default():
        for i, (x_seq, m_seq) in tqdm(enumerate(tf_x_train_miss.take(num_steps)), total=num_steps, desc="Training progress"):
            try:
                with tf.GradientTape() as tape:
                    loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                    losses_train.append(loss.numpy())
                    nll_losses_train.append(nll.numpy())
                    kl_losses_train.append(kl.numpy())
                
                trainable_vars = model.get_trainable_vars()
                grads = tape.gradient(loss, trainable_vars)
                grads = [np.nan_to_num(grad) for grad in grads]
                grads, global_norm = tf.clip_by_global_norm(grads, gradient_clip)
                optimizer.apply_gradients(zip(grads, trainable_vars))

                # Print intermediate results
                if i % print_interval == 0:
                    print("================================================")
                    print("Learning rate: {} | Global gradient norm: {:.2f}".format(optimizer.learning_rate, global_norm))
                    print("Step {}) Time = {:2f}".format(i, time.time() - t0))
                    # loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                    print("Train loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(loss, nll, kl))

                    saver.save(checkpoint_prefix)
                    tf.summary.scalar("loss_train", loss, step=global_step)
                    tf.summary.scalar("kl_train", kl, step=global_step)
                    tf.summary.scalar("nll_train", nll, step=global_step)
                    summary_writer.flush()

                    # Validation loss
                    losses_val_batches = []
                    for x_val_batch, m_val_batch in tf_x_val_miss:
                        # x_val_batch, m_val_batch = tf_x_val_miss.get_next()
                        val_loss, val_nll, val_kl = model.compute_loss(x_val_batch, m_mask=m_val_batch, return_parts=True)
                        losses_val_batches.append([val_loss.numpy(), val_nll.numpy(), val_kl.numpy()])

                    losses_val_batches = np.array(losses_val_batches)
                    # mean across batches
                    avg_losses = np.mean(losses_val_batches, axis=0) 

                    losses_val.append(avg_losses[0])
                    nll_losses_val.append(avg_losses[1])
                    kl_losses_val.append(avg_losses[2])
                    print("Validation loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(avg_losses[0], avg_losses[1], avg_losses[2]))

                    tf.summary.scalar("loss_val", avg_losses[0], step=global_step)
                    tf.summary.scalar("nll_val", avg_losses[1], step=global_step)
                    tf.summary.scalar("kl_val", avg_losses[2], step=global_step)
                    summary_writer.flush()

                    global_step += 1

                    t0 = time.time()
                else:
                    losses_val.append(None)
                    nll_losses_val.append(None)
                    kl_losses_val.append(None)

            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                break

    with open(os.path.join(outdir, "training_curve.tsv"), "w") as outfile:
        data = np.array([losses_train, losses_val, nll_losses_train, nll_losses_val, kl_losses_train, kl_losses_val]).transpose()
        header = ["train_loss", "val_loss", "train_nll", "val_nll", "train_kl", "val_kl"]
        outfile.write("\t".join(header) + "\n")
        for row in data:
            outfile.write("\t".join(map(str, row.tolist())) + "\n")

    
    return model


# incomp_data: (S, V, T), S: samples, V: values, T: time series, NaN values substituted
# with 0.0
def impute(model, incomp_data, inference_batch_size):
    # create batches
    incomp_data_batches = [incomp_data[i: i+inference_batch_size] for i in range(0, len(incomp_data), inference_batch_size)]
    
    batch_recoveries = []

    for batch in tqdm(incomp_data_batches, desc="Imputing progress"):
        # pass the incomp data to the encoder
        encoder_output = model.encode(batch)

        # pass enconder's output most probable latent trajectory to the decoder
        decoder_output = model.decode(encoder_output.mean())

        # most probable data space trajectory
        batch_recovery = decoder_output.mean().numpy()

        batch_recoveries.append(batch_recovery)
    
    recovery = np.concatenate(batch_recoveries, axis=0)

    return recovery


def gpvae_recovery(incomp_data, config_yaml_path, model_checkpoint_path=None, epoch=None, batch_size=None, beta=None, learning_rate=None, sigma=None, length_scale=None, kernel_scales=None, inference_batch_size=None, verbose=True, seed=77):
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
    learning_rate = cfg["optimizer"]["learning_rate"] if learning_rate is None else learning_rate
    sigma = cfg["sigma"] if sigma is None else sigma
    length_scale = cfg["length_scale"] if length_scale is None else length_scale
    kernel_scales = cfg["kernel_scales"] if kernel_scales is None else kernel_scales
    inference_batch_size = cfg["inference_batch_size"] if inference_batch_size is None else inference_batch_size

    # learning rate scheduler
    scheduler_cfg = cfg["optimizer"].get("scheduler", {"type": None})

    #---------------------- Reshape the data to (B,V,T) ------------------------
    recov = recov.transpose().reshape(-1, seq_length, nbr_features)  
    incomp_data = incomp_data.transpose().reshape(-1, seq_length, nbr_features)
    m_mask = m_mask.transpose().reshape(-1, seq_length, nbr_features)

    # incomp_data contains NaN values, set NaN to 0.0
    incomp_data_model = np.nan_to_num(incomp_data, nan=0.0)
    # cast to float32 (net compatibility)
    incomp_data_model = incomp_data_model.astype(np.float32)

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
    if model_checkpoint_path is not None:
        if not os.path.exists(model_checkpoint_path):
            print("Invalid Path to the model checkpoint!")
        else:
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
        start_time_train = time.time()

        if verbose: print("Starting model training...")
        
        outdir = './imputegap_assets/models/' + time.strftime("%Y%m%d_%H%M%S")

        model = train(model, incomp_data_model, m_mask, splits, nbr_features, seq_length, latent_dim, batch_size, epoch, scheduler_cfg, learning_rate, gradient_clip, outdir, verbose=verbose)

        end_time_train = time.time()

        if verbose: print(f"\n> logs: Training gpvae - Execution Time: {(end_time_train - start_time_train):.4f} seconds\n")

        # save a .yaml file containing details about the training
        updated_cfg = copy.deepcopy(cfg)

        updated_cfg["epoch"] = epoch
        updated_cfg["batch_size"] = batch_size
        updated_cfg["optimizer"]["learning_rate"] = learning_rate
        updated_cfg["beta"] = beta
        updated_cfg["sigma"] = sigma
        updated_cfg["length_scale"] = length_scale
        updated_cfg["kernel_scales"] = kernel_scales
        updated_cfg["inference_batch_size"] = inference_batch_size

        updated_cfg["metadata"] = {
            "training_duration_sec": float(np.round(end_time_train - start_time_train, 4))
        }

        with open(os.path.join(outdir, config_yaml_path.split('/')[-1]), 'w') as file:
            yaml.dump(updated_cfg, file, sort_keys=False)
    

    # ---------------- Impute data --------------------------------------
    start_time_impute = time.time()

    if verbose: print("Starting imputation...")

    recovery = impute(model, incomp_data_model, inference_batch_size)

    end_time_impute = time.time()

    if verbose: print(f"\n> logs: Imputing with gpvae - Execution Time: {(end_time_impute - start_time_impute):.4f} seconds\n")

    recov[m_mask] = recovery[m_mask]

    # reshape (B, V, T) -> (T, B*V)
    recov = recov.transpose(2,0,1).reshape(nbr_features, -1)
    
    return recov
