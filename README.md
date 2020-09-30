# RG-Flow: A hierarchical and explainable flow models based on renormalization group and sparse prior
Flow-based generative models have become an important class of unsupervised learning approaches. In this work, we incorporate the key idea of *renormalization group* (RG) and *sparse prior distribution* to design a hierarchical flow-based generative model, called **RG-Flow**, which can separate different scale information of images with disentangle representations at each scale. We demonstrate our method on the CelebA dataset and show that the disentangled representation at different scales enables semantic manipulation and style-mixing of the images. To visualize the latent representation, we introduce the receptive fields for flow-based models and find receptive fields learned by RG-Flow are similar to *convolutional neural networks*. In addition, we replace the widely adopted Gaussian prior distribution by sparse prior distributions to further enhance the disentanglement of representations. From a theoretical perspective, the proposed method has O(log L) complexity for image inpainting compared to previous flow-based models with O(L^2) complexity.

<!--- ## Flow-based generative models --->

## RG-Flow structure

![Image of RG-Flow structure](figs/structure.png)


## Learned receptive fields
![Receptive fields](figs/RF.png)

## Learned factors
![Factors](figs/factors.png)

<!--- * High level factors --->

<!--- **Emotion factor** --->

<!--- ![motion](gifs/smile_video.gif)--->

<!--- **Gender factor**--->

<!--- ![motion](gifs/gender_video.gif)--->

<!--- **Light projection factor**--->

<!--- ![motion](gifs/projection_video.gif)--->

<!--- **Azimuth factor**--->

<!--- ![motion](gifs/rotation_video.gif)--->

<!--- * Mid level factors--->

<!--- * Low level factors--->

## Face-mixing in the scaling direction
![Factors](figs/mix.png)



