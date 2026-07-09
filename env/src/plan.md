flux is actually just a transformer inside a VAE
# TO DO
 - make VAE
 - put flux inside

NEVERMIND

REPA exists

WTF IS THIS

> allowing both VAE
>and diffusion model to be jointly tuned during training pro-
>cess. Through extensive evaluations, we demonstrate that
>end-to-end tuning with REPA-E offers several advantages;
>End-to-End Training Leads to Accelerated Generation
>Performance; speeding up diffusion training by over 17×
>and 45× over REPA and vanilla training recipes


Hmm side note seems like we give clean images to the VAE then add noise to the latent

---

how do REPA-E?

looks like need image encoder -> DinoV3

seems like:
encode image (clean)
interpolate latent with noise
send to dit
pick a random layer in the DiT to apply alignment loss (project first, probably 3x3 padding 1 conv)
get dit output, stop grad and put through batchnorm
do flow matching loss to DiT
do alignment loss to VAE

BRO I WAS READING THE REPA-E DIAGRAM UPSIDE DOWN 😭



hmmmmmmmmmmm need LPIPs and GAN loss for VAE in REPA-E

LPIPS: https://arxiv.org/pdf/1801.03924

L2 loss bad becase slight position shift -> big punishment

I don't want to implement VGG so will just use https://github.com/richzhang/perceptualsimilarity

GAN Loss: https://arxiv.org/pdf/1512.09300


need to do:

- cfg
- ema

but first: kl divergence loss is broken
i think at first the alignment loss was never getting backpropped
because i did zero grad after its backward
but now that the zero grad is before the training step, the alignment is actually doing something
and it is causing kl to spike

- maybe zero-init conv projection?


HOLY I AM STUPID I WASNT PASSING THE NOISED LATENTS INTO THE DIT