# Generative Models: SpyGAN
[Pytorch implementation](https://github.com/ahirner/generative-models/tree/master/GAN/spy_gan) with collaborative weight sharing between D and G. Extended from Vanilla GAN in the collection of generative models by [wiseodd][https://github.com/wiseodd/generative-models].

## Results
![alt-text](https://raw.githubusercontent.com/ahirner/generative-models/master/GAN/spy_gan/SpyGAN.png)
(SpyGAN converges faster than Vanilla GAN, all things being equal)

## Notes
Slow convergence (and mode collapse) is a common problem with GANs because many local minima exist in a zero-sum game. Also, there are few attempts to introduce collaboration in the form of message passing/weight sharing. For example, [CoGAN](https://arxiv.org/abs/1606.07536) shares weights within a population of disparate Gs and Ds which doesn't change the reward distribution of the game.

This variant introduces know-how sharing by enforcing D to asisst G. In each iteration, G's generative layer absorbes a fraction of the transpose of D's discriminatory layer. Thus, D's initial advantage of a low-dimensional and certain signal from the true distribution is passed onto G.

### Exploration
- Other transfer functions than linear, e.g. drop-out [seemed to aggrevate mode collapse]
- Vary intensity of weight transfer depending on advantage of D over G (LG/LD) [constant transfer seems let G diverge over time]
- Sharing of convolution kernels instead of FC
- How does such a transfer affect diversity?
- Overcome local minima by additional loss term of joint objective (LG*LD?)
