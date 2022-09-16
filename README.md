# Stylegan toy case
Training code for stylegan toy case with customized loss terms

# Details
1. Training data: 1-D manifold helix dataset in a 3D space
2. Z space: 2D gaussian distribution
3. W space: KL regularization implemented to "pump the distribution ball"

# Sample output:
  This result is without the "pumping ball", i.e. lambda = 0 for KL-reg term 
<p align="center">
  <img src="https://github.com/GuangyuNie/stylegan_toy_case/blob/master/sample_image/latent_9900.png" width="350" title="latent distribution">
  <img src="https://github.com/GuangyuNie/stylegan_toy_case/blob/master/sample_image/output_9990.png" width="350" alt="output distribution">
</p>
<p align="center">
latent distribution &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; output distribution
</p>
# Note
1. Hyperparameter not carefully tuned, the training may takes excessive time'
2. Please check envionment.yml for env
