# New experiment

I have trained a model on HMNIST, using the missingness provided by the authors, that was a MNAR missingness, where white pixels had two times the probability of black pixels to be missing.

Given the author's mask I have computed the rate of missingness, that is around 45% for each sample (10 images), and also across images.

With this experiment I wanted to verify how well a GP-VAE model adapts to data that don't have the same missingness patterns. To do so I have used ImputeGAP to contaminate HMNIST dataset in two ways, using a Scattered and MCAR (M Completelly At Random) contamination. Then imputed a subset of 120 samples from the dataset using the trained model and computed NLL and MSE on the imputations.

Results on MNAR (authors' miss):

```python
{'nll': 0.35284719920547486, 'mse': 0.11048427755163014}
((0.44792572766811645, 0.022531871365504545), (0.4479257276681164, 0.015794818474347896), 0.44792572766811645, (4.479257276681165, 1.7810154315420224))
```

Since the contamination with scattered has an offset where at the beginning of the series there is no missingness, I have contaminated 120 samples, but then NLL and MSE is compute on a single sample in the middle.

Results on Scattered:

The starting position is randomly shifted by adding a random value to W (start offset), then progresses until the size of the missing block is reached, affecting the first series from the top up to S% of the dataset.

```python
rate_dataset = 1.0
rate_series = 0.4
offset=0.1
{'nll': 0.3914378550577921, 'mse': 0.11196729045447397}
((0.8110969387755101, 0.00175353661796781), (0.8110969387755103, 0.0), 0.8110969387755103, (8.110969387755102, 3.8447794022705795))

rate_series = 0.3
{'nll': 0.1633589169987756, 'mse': 0.0604566683964712}
((0.4915816326530612, 0.0010204081632653197), (0.4915816326530612, 0.0), 0.4915816326530612, (4.915816326530612, 4.951350552404331))
```

Results on MCAR:

Data blocks of the same size are removed from arbitrary series at a random position between W and N, until the total number of missing values per series is reached.

```python
rate_dataset = 1.0
rate_series = 0.4
block_size = 3
{'nll': 0.1683910375253438, 'mse': 0.06403385049365304}
((0.4521683673469387, 0.017941219708808357), (0.45216836734693877, 0.0), 0.45216836734693877, (4.521683673469388, 2.8863753171349873))

rate_series = 0.5
{'nll': 0.16073697408040363, 'mse': 0.06319444444444444}
((0.5510204081632653, 0.011322622969985784), (0.5510204081632653, 0.0), 0.5510204081632653, (5.510204081632653, 2.9636226382628674))

rate_series = 0.6
{'nll': 0.1724155533502388, 'mse': 0.07140100347356233}
((0.6609693877551021, 0.01096938775510202), (0.6609693877551021, 0.0), 0.6609693877551021, (6.60969387755102, 2.900247686143879))

rate_series = 0.8
{'nll': 0.5962941949621536, 'mse': 0.13838903170522707}
((0.8931122448979592, 0.007981881554674487), (0.8931122448979592, 0.0), 0.8931122448979592, (8.931122448979592, 1.9067149774977934))
```
