# New experiment (reviewed)

I have trained a model on HMNIST, using the missingness provided by the authors, that was a MNAR missingness, where white pixels had two times the probability of black pixels to be missing.

Given the author's mask I have computed the rate of missingness of 
a general pixel and of a white pixel. For a general pixel is around 45% 
for each sample (10 images), and also across sample images, 
while for a white pixel is around 80% for each sample and across 
images.

With this experiment I wanted to verify how well a GP-VAE model adapts 
to data that don't have the same missingness patterns. 
To do so I have used ImputeGAP to contaminate HMNIST dataset in two 
ways, using a Scattered and MCAR (Missing Completely At Random) 
contamination. For the contamination I have selected the first 
122 samples of the HMNIST validation dataset, the reason is that 
the these methods have a starting offset where no missingness is added, 
and this is something it does not happen with the MNAR provided by the 
authors, therefore for comparability I have contaminated on 122 samples, 
and evaluated NLL and MSE metrics as well as missingness ratio only on 
the 61th sample.

Results on MNAR (authors' miss):

```python
{'nll': 0.41146656504848544, 'mse': 0.14035087719298245}
((0.44792572766811645, 0.022531871365504545), (0.4479257276681164, 0.015794818474347896), 0.44792572766811645)
((0.7985515422503437, 0.04455509156589401), (0.7985603322506796, 0.014754206950447825), 0.7980383023229243)
```

Results on Scattered:

The starting position is randomly shifted by adding a random value to W (start offset), then progresses until the size of the missing block is reached, affecting the first series from the top up to S% of the dataset.

```python
rate_dataset = 1.0
rate_series = 0.4
offset=0.1
{'nll': 0.3802622881938984, 'mse': 0.10960842899827017}
((0.8110969387755101, 0.00175353661796781), (0.8110969387755103, 0.0), 0.8110969387755103)
((0.8318665395389702, 0.03292150449257605), (0.8319017564088231, 0.01894916417498847), 0.8316473162885946)

rate_series = 0.3
{'nll': 0.16510991649301052, 'mse': 0.06123508043591074}
((0.4915816326530612, 0.0010204081632653197), (0.4915816326530612, 0.0), 0.4915816326530612)
((0.4787988585973738, 0.048584464189767314), (0.4787624715616694, 0.02634185841406962), 0.4776364707019749)
```

Results on MCAR:

Data blocks of the same size are removed from arbitrary series at a random position between W and N, until the total number of missing values per series is reached.

```python
rate_dataset = 1.0
rate_series = 0.4
block_size = 3
{'nll': 0.17694811477983957, 'mse': 0.06770098730606489}
((0.4521683673469387, 0.017941219708808357), (0.45216836734693877, 0.0), 0.45216836734693877)
((0.45478103506463297, 0.051001401070786476), (0.4547431813075219, 0.019932446910519225), 0.45480273161128154)

rate_series = 0.5
{'nll': 0.15755174424913193, 'mse': 0.06643518518518518}
((0.5510204081632653, 0.011322622969985784), (0.5510204081632653, 0.0), 0.5510204081632653)
((0.5475742804865624, 0.04897665453875182), (0.5475618010699037, 0.018605401354489237), 0.5485898100703996)

rate_series = 0.6
{'nll': 0.1823709738940081, 'mse': 0.07352373600926283}
((0.6609693877551021, 0.01096938775510202), (0.6609693877551021, 0.0), 0.6609693877551021)
((0.6525635958455327, 0.048000821856506296), (0.6525648317871828, 0.016971061348416742), 0.6545847652027176)

rate_series = 0.8
{'nll': 0.6195778605666238, 'mse': 0.13867466438160525}
((0.8931122448979592, 0.007981881554674487), (0.8931122448979592, 0.0), 0.8931122448979592)
((0.8831575011187768, 0.035708364036102266), (0.8831874080577712, 0.013445087401420558), 0.8840383550567328)
```
