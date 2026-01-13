import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# for computer vision datasets (HMNIST and SPRITES) imputation visualization
def create_imputation_plot(image_shape, time_length, miss, imputed_no_gt, imputed, gt, sample_id, figsize=(15,5), cmap="gray"):
  miss[np.isnan(miss)] = 0.0
  fig, axes = plt.subplots(4, time_length, figsize=figsize, layout='constrained', sharey=True)

  for j in range(time_length):
    axes[0, j].imshow(np.clip(miss[sample_id,j].reshape(*image_shape), 0, 1), cmap=cmap)
    axes[1, j].imshow(np.clip(imputed_no_gt[sample_id,j].reshape(*image_shape), 0, 1), cmap=cmap)
    axes[2, j].imshow(np.clip(imputed[sample_id,j].reshape(*image_shape), 0, 1), cmap=cmap)
    axes[3, j].imshow(np.clip(gt[sample_id,j].reshape(*image_shape), 0, 1), cmap=cmap)
    axes[0,j].axis('off')
    axes[1,j].axis('off')
    axes[2,j].axis('off')
    axes[3,j].axis('off')
  axes[0,0].set_ylabel("Miss")
  fig.suptitle("Miss -> Imputed (no GT) -> Imputed | GT")
  plt.show()


# mask given as (S, V, T)
def compute_mask_statistics(mask):
  s, v, t = mask.shape
  miss_per_img = np.sum(mask, axis=2).reshape(-1) / t
  img_avg, img_std = np.mean(miss_per_img), np.std(miss_per_img)

  miss_per_sample = np.sum(mask, axis=(2,1)) / (v*t)
  sample_avg, sample_std = np.mean(miss_per_sample), np.std(miss_per_sample)

  tot_miss_ratio = np.sum(mask) / (s*v*t)

  mask = mask.transpose(0,2,1)
  miss_per_series = np.sum(mask, axis=2)
  series_avg = np.mean(miss_per_series, axis=1)
  series_std = np.std(miss_per_series, axis=1)

  return ((img_avg, img_std), (sample_avg, sample_std), tot_miss_ratio, (np.mean(series_avg), np.mean(series_std)))

def compute_mask_statistics_v2(mask, values=None, target_value=None):
    """
    mask: binary array (1 = missing), shape (s, v, t)
    values: original data array, same shape as mask
    target_value: value to condition missingness on (e.g. 1 for white pixels)
    """

    s, v, t = mask.shape

    # If conditioning on a value
    if values is not None and target_value is not None:
        value_mask = (values == target_value)
        effective_mask = mask & value_mask
        denom = np.sum(value_mask)
    else:
        effective_mask = mask
        denom = s * v * t

    # ---- Per-image missingness ----
    miss_per_img = np.sum(effective_mask, axis=2).reshape(-1)
    img_denom = np.sum(value_mask, axis=2).reshape(-1) if values is not None and target_value is not None else t
    miss_per_img = np.divide(miss_per_img, img_denom, where=img_denom > 0)

    img_avg, img_std = np.mean(miss_per_img), np.std(miss_per_img)

    # ---- Per-sample missingness ----
    miss_per_sample = np.sum(effective_mask, axis=(1, 2))
    sample_denom = np.sum(value_mask, axis=(1, 2)) if values is not None and target_value is not None else v * t
    miss_per_sample = np.divide(miss_per_sample, sample_denom, where=sample_denom > 0)

    sample_avg, sample_std = np.mean(miss_per_sample), np.std(miss_per_sample)

    # ---- Total missingness ----
    tot_miss_ratio = np.sum(effective_mask) / denom if denom > 0 else 0.0

    return (
        (img_avg, img_std),
        (sample_avg, sample_std),
        tot_miss_ratio
    )


def plot_training_results(path):
  with open(os.path.join(path, 'training_curve.tsv'), 'r') as infile:
      df = pd.read_csv(infile, sep='\t')

  # Filter to fetch only rows where also validation losses were computed
  no_nan = df.loc[np.invert(df['val_loss'].isna())]

  _, axes = plt.subplots(1, 3, figsize=(10, 4), layout='constrained')
  axes[0].plot(df.index, df['train_loss'], label='Train')
  axes[0].plot(no_nan.index, no_nan['val_loss'], label='Validation')
  axes[0].legend(loc='upper right')
  axes[0].set_title("Train and validation loss")
  axes[0].set_ylabel("Loss: NLL + beta*KL")
  axes[0].set_xlabel("Training steps")
  axes[1].plot(df.index, df['train_nll'], label='Train')
  axes[1].plot(no_nan.index, no_nan['val_nll'], label='Validation')
  axes[1].legend(loc='upper right')
  axes[1].set_title("Train and validation NLL")
  axes[1].set_ylabel("NLL")
  axes[1].set_xlabel("Training steps")
  axes[2].plot(df.index, df['train_kl'], label="Train")
  axes[2].plot(no_nan.index, no_nan['val_kl'], label="Validation")
  axes[2].legend(loc='upper right')
  axes[2].set_title("Train and validation KL")
  axes[2].set_ylabel("KL")
  axes[2].set_xlabel("Training steps")
  plt.show()
