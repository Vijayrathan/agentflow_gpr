Your pipeline would be:

Simulation Parameters→XGBoost→Predicted PCA Latents→Inverse PCA→Predicted Signal

# Parameter → Signal Surrogate Pipeline

$$
\text{Simulation Parameters} \rightarrow \text{XGBoost} \rightarrow \text{Predicted PCA Latents} \rightarrow \text{Inverse PCA} \rightarrow \text{Predicted Signal}
$$

## Training

$$
X = \text{simulation parameters}
$$

$$
Y = \text{PCA latent coefficients of the true signal}
$$

For each simulation sample:

$$
x_i \rightarrow z_i
$$

where $z_i$ is obtained by applying PCA to that sample's true EM signal.

XGBoost learns:

$$
f(x_i) \approx z_i
$$

## Inference

For a new set of parameters:

$$
x_{\text{new}} \rightarrow \hat z
$$

and you reconstruct:

$$
\hat y = PCA^{-1}(\hat z)
$$

where $\hat y$ is the predicted full signal.

## Separate Electric and Magnetic PCA Spaces

$$
x \rightarrow \text{XGBoost} \rightarrow [\hat z_E, \hat z_H]
$$

then:

$$
\hat{\mathbf E} = PCA_E^{-1}(\hat z_E)
$$

$$
\hat{\mathbf H} = PCA_H^{-1}(\hat z_H)
$$

giving back $\hat E_x, \hat E_y, \hat E_z, \hat H_x, \hat H_y, \hat H_z$ — each with length **5282**.

## Key Distinction

> **PCA learns how to represent the signal.**
>
> **XGBoost learns how simulation parameters map to that representation.**

The PCA latent vectors become the training targets / ground-truth outputs for XGBoost, and inverse PCA converts XGBoost's latent predictions back into the physical signal.


## Future steps

Once the model is trained. Perform saliency -> Find out which region needs more data -> send the feedback to the data generator -> Retrain... 









-----



The paper also used sythetic data with gprmax,they ran the sim files one by one with changing concrete water fraction, rebar radius and rebar depth.

They reduced 3000 dim to 300  then PCA

Their total N = 2000

All 2000 has fixed domain, fixed grid and fixed antenna setup

For me :

Undersample the 5282 to 300 then perform PCA -- NOT NEEDED. PCA is capable of handling 5000 or 8000 dim also, downsampling is lossy

Total target 2000

I vary layer thickness, bd, pd, cylinder radius

Both of us are doing random sampling


The main difference is my data generation engine. Mine is agentic engine, which can generate any scenario at any scale possibly using LHS/Sobol/Monte Carlo