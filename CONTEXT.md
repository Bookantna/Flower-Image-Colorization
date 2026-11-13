# Flower Image Colorization — Glossary

Terms used in the notebook, README, and presentation. One meaning per term.

## Dataset

- **Flower dataset** — The `flower_dataset.zip` archive downloaded in the notebook. 128×128 RGBA PNGs of hand-drawn (synthetic) flowers on a transparent background.
- **Species** — The flower type, the first part of a filename. Valid species: `azalea`, `daisy`, `marigold`, `orchid`, `rose` (five flowers).
- **Angle** — The number after the species (`0`–`359`). The flower head's rotation in degrees. Adjacent angles produce near-identical images.
- **Stem** — `core1`–`core6`. One of six stem shapes drawn under the same flower head. The README's "6 stems" refers to these.
- **Flip** — The `_flip` suffix. A horizontal mirror of the same image.
- **Variant** — One image, identified by *(species, angle, stem, flip)*. The full set is 5 species × 360 angles × 6 stems × 2 flips = **21,600** variants.
- **Duplicate files** — `sunflower_*` and `.DS_S_*` files are byte-identical copies of `rose` images (a broken export, not a sixth species). They are excluded. `.DS_Store` is macOS metadata and is also excluded.
- **Subset** — The random, species-stratified fraction (`SUBSET_FRACTION`, default 20%) of variants actually used, so training fits a Colab session.
- **Folded angle** — `min(angle, 360 − angle)`, in 0–180°. A flipped image at angle *a* resembles an unflipped one at *360 − a*, so mirror twins share a folded angle.
- **Angle block** — A contiguous range of folded angles (`ANGLE_BLOCK` degrees wide). The unit used for the train/test split. Training images within `SPLIT_GAP` degrees of any test angle are dropped, so test images have no near-rotation or mirror twin in training.

## Model

- **Gray input** — The generator's input: luminance `0.299R + 0.587G + 0.114B` of the colour image, after compositing onto white.
- **Generator (G)** — U-Net that maps a gray input (1 channel) to a colour image (3 channels, `tanh` range [-1, 1]).
- **Discriminator (D)** — PatchGAN. Scores *(gray input, colour image)* pairs per patch as real or fake. Outputs logits, with no sigmoid.
- **Adversarial loss** — `BCEWithLogitsLoss` on D's patch logits.
- **L1 loss** — Mean absolute error between generated and real colour. Weighted by `L1_LAMBDA` (pix2pix default 100) in the generator objective.
- **CGAN** — In this project, the pix2pix-style conditional GAN: G and D are both conditioned on the gray input.

## Evaluation

- **Test set** — Variants whose angle block was held out of training.
- **PSNR / SSIM** — Per-image fidelity metrics between generated and real colour in [0, 1], averaged over the test set.
