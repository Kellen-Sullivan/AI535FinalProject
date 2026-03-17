# Ultralytics YOLO Built-In Augmentation Settings Guide

All of these settings can be passed as keyword arguments to `model.train()`. Each value controls the **intensity** or **probability** of that augmentation being applied during training.

---

## Color & Lighting Augmentations

### `hsv_h` — Hue Shift

- **Type:** `float` | **Default:** `0.015` | **Range:** `0.0 – 1.0`
- Shifts the hue channel of the image by a fraction of the full color wheel (0–360°).
- A value of `0.015` means the hue can shift by up to ±1.5% of the wheel (~±5.4°).
- `0.0` = no hue change, `1.0` = hue can shift across the entire spectrum.
- **Use case:** Helps the model generalize across different lighting conditions and color casts (e.g., warm vs. cool light).

### `hsv_s` — Saturation Shift

- **Type:** `float` | **Default:** `0.7` | **Range:** `0.0 – 1.0`
- Randomly adjusts the saturation (color intensity) of the image.
- `0.7` means saturation can vary by up to ±70%.
- `0.0` = no saturation change, `1.0` = maximum saturation variation.
- **Use case:** Simulates faded, washed-out, or highly saturated environments.

### `hsv_v` — Brightness (Value) Shift

- **Type:** `float` | **Default:** `0.4` | **Range:** `0.0 – 1.0`
- Randomly adjusts the brightness of the image.
- `0.4` means brightness can vary by up to ±40%.
- `0.0` = no brightness change, `1.0` = maximum brightness variation.
- **Use case:** Helps the model handle shadows, overexposure, and varying ambient light.

---

## Geometric / Spatial Augmentations

### `degrees` — Rotation

- **Type:** `float` | **Default:** `0.0` | **Range:** `0.0 – 180.0`
- Randomly rotates the image within ±N degrees.
- `0.0` = no rotation, `15.0` = images may rotate up to ±15°.
- **Use case:** Helps detect objects at tilted or angled orientations. Useful for aerial/drone imagery.

### `translate` — Translation

- **Type:** `float` | **Default:** `0.1` | **Range:** `0.0 – 1.0`
- Shifts the image horizontally and vertically by a fraction of the image size.
- `0.1` = images shift up to ±10% of their width/height.
- `0.0` = no translation, `1.0` = images can shift up to 100%.
- **Use case:** Teaches the model to detect partially visible or off-center objects.

### `scale` — Scaling

- **Type:** `float` | **Default:** `0.5` | **Range:** `0.0 – 1.0`
- Randomly scales (zooms in/out) the image by a gain factor.
- `0.5` means the image may be scaled between 0.5× and 1.5× its original size.
- `0.0` = no scaling, `1.0` = aggressive zoom in/out.
- **Use case:** Simulates objects at different distances from the camera.

### `shear` — Shear

- **Type:** `float` | **Default:** `0.0` | **Range:** `-180.0 – 180.0`
- Applies a shearing transformation by the specified degrees.
- `0.0` = no shear.
- **Use case:** Mimics objects viewed from oblique angles.

### `perspective` — Perspective Transform

- **Type:** `float` | **Default:** `0.0` | **Range:** `0.0 – 0.001`
- Applies a random perspective warp. Values are very small because even tiny amounts create noticeable distortion.
- `0.0` = no perspective change, `0.001` = maximum perspective warp.
- **Use case:** Helps the model understand 3D perspective effects (e.g., objects viewed from above/below).

---

## Flip Augmentations

### `flipud` — Vertical Flip (Up-Down)

- **Type:** `float` | **Default:** `0.0` | **Range:** `0.0 – 1.0`
- **This is a probability.** Each image has this chance of being flipped vertically.
- `0.0` = never flipped, `0.5` = 50% chance, `1.0` = always flipped.
- **Use case:** Useful for satellite/aerial imagery or microscopy where "up" is arbitrary. Usually left at `0.0` for standard photos since most real-world scenes have a consistent vertical orientation.

### `fliplr` — Horizontal Flip (Left-Right)

- **Type:** `float` | **Default:** `0.5` | **Range:** `0.0 – 1.0`
- **This is a probability.** Each image has this chance of being flipped horizontally.
- `0.0` = never flipped, `0.5` = 50% chance, `1.0` = always flipped.
- **Use case:** One of the most universally useful augmentations. Most objects look the same mirrored left-right. Set to `0.0` if direction matters (e.g., reading text).

### `bgr` — BGR Channel Swap

- **Type:** `float` | **Default:** `0.0` | **Range:** `0.0 – 1.0`
- **This is a probability.** Swaps image channels from RGB to BGR order.
- `0.0` = never swapped, `1.0` = always swapped.
- **Use case:** Guards against incorrect channel ordering in input data. Not needed if your pipeline is consistent.

---

## Advanced Composition Augmentations

### `mosaic` — Mosaic

- **Type:** `float` | **Default:** `1.0` | **Range:** `0.0 – 1.0`
- **This is a probability.** Combines 4 training images into a single 2×2 grid image.
- `1.0` = always applied, `0.0` = disabled.
- **Use case:** One of YOLO's most powerful augmentations. Forces the model to see multiple scenes and object scales in a single image. Greatly improves small-object detection and scene understanding.
- **Note:** Typically disabled in the final few epochs via `close_mosaic` to stabilize training.

### `mixup` — MixUp

- **Type:** `float` | **Default:** `0.0` | **Range:** `0.0 – 1.0`
- **This is a probability.** Blends two images together (and their labels) by overlaying them with transparency.
- `0.0` = disabled, `0.5` = 50% chance per image.
- **Use case:** Introduces label noise and visual variety, helping the model generalize and avoid overconfident predictions.

### `cutmix` — CutMix

- **Type:** `float` | **Default:** `0.0` | **Range:** `0.0 – 1.0`
- **This is a probability.** Cuts a rectangular region from one image and pastes it onto another, blending their labels proportionally.
- `0.0` = disabled.
- **Use case:** Creates natural occlusion scenarios, improving robustness to partially hidden objects.

### `copy_paste` — Copy-Paste (Segmentation Only)

- **Type:** `float` | **Default:** `0.0` | **Range:** `0.0 – 1.0`
- **This is a probability.** Copies object instances from one image and pastes them into another.
- `0.0` = disabled.
- **Use case:** Increases the number and variety of object instances per image. Only works with segmentation tasks since it requires instance masks.

### `copy_paste_mode` — Copy-Paste Strategy

- **Type:** `str` | **Default:** `"flip"` | **Options:** `"flip"`, `"mixup"`
- Controls how copy-paste augmentation works:
  - `"flip"` — pastes a flipped version of the copied objects.
  - `"mixup"` — blends copied objects with the target image.

---

## Classification-Only Augmentations

These only apply when training classification models (not detection/segmentation).

### `auto_augment` — Auto Augmentation Policy

- **Type:** `str` | **Default:** `"randaugment"` | **Options:** `"randaugment"`, `"autoaugment"`, `"augmix"`
- Applies a predefined augmentation policy:
  - `"randaugment"` — randomly applies N augmentations from a set at random magnitudes.
  - `"autoaugment"` — uses a learned augmentation policy (from the AutoAugment paper).
  - `"augmix"` — mixes multiple augmentation chains together for robustness.

### `erasing` — Random Erasing

- **Type:** `float` | **Default:** `0.4` | **Range:** `0.0 – 1.0`
- **This is a probability.** Randomly erases a rectangular region of the image.
- `0.4` = 40% chance per image, `0.0` = disabled.
- **Use case:** Forces the model to focus on multiple features rather than relying on a single discriminative region.

---

## Custom Augmentations

### `augmentations` — Albumentations Transforms (Python API Only)

- **Type:** `list` | **Default:** `[]`
- Pass a list of Albumentations transform objects for additional augmentations beyond the built-in ones.
- These are applied **on top of** the built-in augmentations above.
- Example:
  ```python
  import albumentations as A

  model.train(
      data="dataset.yaml",
      augmentations=[
          A.CLAHE(clip_limit=2.0, p=0.5),
          A.GaussianBlur(blur_limit=(3, 7), p=0.3),
      ],
  )
  ```

---

## Related Training Setting

### `close_mosaic` — Disable Mosaic in Final Epochs

- **Type:** `int` | **Default:** `10`
- Turns off mosaic augmentation for the last N epochs of training.
- Setting to `0` keeps mosaic on for the entire training.
- **Use case:** Mosaic is great for learning but can hurt fine-tuning in the final stages. Disabling it at the end lets the model stabilize on "normal" single images before finishing.

---

## Quick Reference Table

| Argument          | Type  |Default| Range     | What the Value Means |
|-------------------|-------|-------|-----------|----------------------|
| `hsv_h`           | float | 0.015 | 0.0–1.0   | **Intensity** — fraction of hue wheel shift |
| `hsv_s`           | float | 0.7   | 0.0–1.0   | **Intensity** — fraction of saturation change |
| `hsv_v`           | float | 0.4   | 0.0–1.0   | **Intensity** — fraction of brightness change |
| `degrees`         | float | 0.0   | 0–180     | **Max degrees** of rotation |
| `translate`       | float | 0.1   | 0.0–1.0   | **Fraction** of image size to shift |
| `scale`           | float | 0.5   | 0–1       | **Gain factor** for zoom in/out |
| `shear`           | float | 0.0   | -180–180  | **Degrees** of shear |
| `perspective`     | float | 0.0   | 0.0–0.001 | **Warp coefficient** (tiny values = big effect) |
| `flipud`          | float | 0.0   | 0.0–1.0   | **Probability** of vertical flip |
| `fliplr`          | float | 0.5   | 0.0–1.0   | **Probability** of horizontal flip |
| `bgr`             | float | 0.0   | 0.0–1.0   | **Probability** of channel swap |
| `mosaic`          | float | 1.0   | 0.0–1.0   | **Probability** of 4-image mosaic |
| `mixup`           | float | 0.0   | 0.0–1.0   | **Probability** of image blending |
| `cutmix`          | float | 0.0   | 0.0–1.0   | **Probability** of cut-and-paste blending |
| `copy_paste`      | float | 0.0   | 0.0–1.0   | **Probability** of instance copy-paste |
| `copy_paste_mode` | str   | "flip"| —         | **Strategy** for copy-paste |
| `erasing`         | float | 0.4   | 0.0–1.0   | **Probability** of random erasing (classify only) |
| `augmentations`   | list  | []    | —         | **List** of Albumentations transforms |
| `close_mosaic`    | int   | 10    | 0+        | **Epochs** before end to disable mosaic |
