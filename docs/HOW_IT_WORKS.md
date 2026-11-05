# How it works: a walkthrough of the code

This is the guide to read before an interview. Each section answers a question an interviewer is likely to ask.

## 1. Why CycleGAN and not a normal supervised model?

A supervised colorizer (for example a U-Net with an L1 loss against the true colors) needs exact grayscale/color pairs. For cartoons we can make those pairs by desaturating, but a plain L1 loss tends to give dull, brownish "average" colors, because averaging across all plausible colors minimizes the error. A GAN loss rewards colors that *look* real instead, so results are more vivid. CycleGAN adds cycle consistency so that it still works without pairs.

**Follow-up you might get:** *"Since you can make pairs, why not pix2pix?"* That's a fair point. Pix2pix (conditional GAN + L1 on pairs) is a strong baseline here. CycleGAN's advantage is that it also works when the grayscale data comes from a different source than the color data, like real old black-and-white cartoons rather than desaturated modern ones.

## 2. The four networks (`cyclegan/models.py`)

- `G`: grayscale → color. `F`: color → grayscale. Both use the same `Generator` class.
- `D_A` judges whether a grayscale image is real, and `D_B` does the same for color images. Both use the `Discriminator` class.

**Generator.** A 7×7 conv, then two stride-2 convs that downsample to ¼ resolution, then 9 residual blocks that do the "thinking" at low resolution, then two transposed convs back up and a final 7×7 conv with `tanh`. Reflection padding avoids dark borders. Instance norm (normalizing each image on its own) works better than batch norm for style transfer and with batch size 1.

**PatchGAN discriminator.** It doesn't output one real/fake number. It outputs a grid (30×30 for a 256 px input), where each cell judges a ~70×70 patch. Local realism is what matters for color and texture, and it needs fewer parameters.

## 3. The losses (`train.py`)

```
loss_G = adversarial + 10 · cycle + 5 · identity
```

- **Adversarial:** least-squares (`MSELoss`) toward 1 for fakes. LSGAN gives smoother gradients than the original log loss, so training is more stable.
- **Cycle:** `|F(G(a)) − a| + |G(F(b)) − b|`. Without it, G could output any nice color cartoon, unrelated to the input. With it, the output has to keep enough information to recover the input.
- **Identity:** `|G(b) − b|`. Giving G an image that's already colored shouldn't change it. This keeps G from shifting the whole palette.

Each discriminator's loss is `0.5 · (MSE(D(real), 1) + MSE(D(fake), 0))`. The 0.5 slows D down relative to G.

## 4. Training tricks

- **Image buffer (`ImageBuffer`):** D sees a mix of the newest fakes and older ones (50 stored). This keeps G and D from oscillating.
- **LR schedule:** Adam (lr 2e-4, β₁ = 0.5) held constant, then decayed linearly to 0 over the second half of training.
- **`fake.detach()` in the D step:** stops the discriminator's gradients from flowing back into the generators.
- **Init:** conv weights drawn from N(0, 0.02), as in the paper.

## 5. Data (`cyclegan/data.py`)

`UnpairedColorizationDataset` shuffles the frames with a fixed seed and splits them in half. Half A goes through `to_gray3`, which makes the image grayscale but keeps 3 channels so both domains have the same tensor shape. Half B stays in color. Each A sample is paired with a *random* B sample, which is fine because training is unpaired. Images are scaled to [-1, 1] to match `tanh`.

## 6. Inference (`colorize.py`)

Only `G` is needed. The checkpoint `colorizer.pt` stores just G's weights plus the image size. For video, each frame is converted to grayscale, resized to the training size, colorized, resized back and written out. The script times the whole loop and prints the FPS, which is the number to quote for "real-time".

## 7. Questions to be ready for

- *How do you evaluate a GAN?* Look at sample grids, then compute FID between colorized frames and held-out real color frames. Lower FID means closer distributions.
- *What failure modes did you see?* Typical ones are color bleeding across line art, flickering between video frames because each frame is colored independently, and mode collapse where everything gets one palette.
- *How would you reduce flicker in video?* Add a temporal consistency loss between neighboring frames, or condition on the previous colorized frame.
