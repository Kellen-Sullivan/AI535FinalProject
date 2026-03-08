# Project Development Notes

*Comparing Baseline model (no augmentation) and improved model (with augmentation during training); both trained on degraded validation sets*

Baseline Model
1. train baseline model (normally)
2. generate degraded validation sets (copies of validation images, each w/ different degradation levels) --> used as test conditions, not training data
3. evaluate baseline model on each set (measures robustness)

Robustness-Augmented Model
* add degradation during training by applying random augmentations per training image
* every epoch, model sees slightly degraded images

clean image -> random degradation applied -> training input

* evaluate model on same degraded validation set



