from ultralytics import YOLO
from ultralytics.data.augment import Albumentations, UnderwaterColorAttenuation, UnderwaterHaze
from IPython.display import Image
from roboflow import Roboflow
import wandb
import argparse
# from augmentations import UnderwaterColorAttenuation, UnderwaterHaze
import albumentations as A

rf = Roboflow(api_key="fZeOJWIPyFVGlTI0Q5lj")
project = rf.workspace("kellens-workspace-ausjh").project("underwater-trash-segmentation-io1hv")
version = project.version(2)
dataset = version.download("yolov11")

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Special Augmentations need to be setup as functions in the ../ultralytics/data/augment.py file, and then added to the SPECIAL_AUG_MAP above
# Unless you're using them, you can ignore the code below. It's just a way to specify custom augmentations that aren't already built into the YOLO training pipeline.
SPECIAL_AUG_MAP = {
    "color_attenuation": UnderwaterColorAttenuation,
    "haze": UnderwaterHaze,
}

def populate_special_augs(special_aug_names):
    special_augs = []
    for aug in special_aug_names:
        if aug in SPECIAL_AUG_MAP:
            special_augs.append(SPECIAL_AUG_MAP[aug]())
        else:
            print(f"Warning: Unknown augmentation '{aug}' specified. Skipping.")
    return special_augs
#------------------------------------------------------------------------------------------------------------------------------------------------------

def on_fit_epoch_end(trainer):
    epoch = trainer.epoch + 1
    wandb.log({"epoch": epoch, **trainer.metrics})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_augmentations", nargs='*', default=[], help="List of strings specifying which augmentations to apply during training.")
    parser.add_argument("--special_augmentations", nargs='*', default=[], help="List of strings specifying which augmentations to apply during training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="The optimizer to use for training (e.g., 'Adam', 'SGD').")
    
    parser.add_argument("--wandb_group", type=str, default="unnamed", help="The group name for the Weights & Biases run.")
    parser.add_argument("--wandb_name", type=str, default="unnamed", help="The name of the Weights & Biases run.")
    args = parser.parse_args()

    data_yaml = f"{dataset.location}/data.yaml"

    wandb.init(
        project="yolo11-trash-seg", 
        group=args.wandb_group, 
        name=args.wandb_name, 
        config={
            "epochs": args.epochs,
            "dataset": data_yaml,
            "model": "yolo11n-seg.pt",
            "special_augmentations": args.special_augmentations,
            "yolo_augmentations": args.yolo_augmentations,
            "optimizer": args.optimizer
        })

    # Special Augmentations need to be setup as functions in the ../ultralytics/data/augment.py file, and then added to the SPECIAL_AUG_MAP above
    special_augs = []
    if len(args.special_augmentations) > 0:
        special_augs = populate_special_augs(args.special_augmentations)
    print(f"Using special augmentations: {[type(aug).__name__ for aug in special_augs]}")

    # Load the seg model
    model = YOLO("yolo11m-seg.pt")
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    yolo_augs = {k: float(v) for k, v in (s.split('=') for s in args.yolo_augmentations)}
    # Train model on custom dataset
    if len(special_augs) > 0:
        results = model.train(task='segment', mode='train', data=data_yaml, epochs=args.epochs, optimizer=args.optimizer, **yolo_augs, augmentations=special_augs)
    else:
        results = model.train(task='segment', mode='train', data=data_yaml, epochs=args.epochs, optimizer=args.optimizer, **yolo_augs)

    test_results = model.val(data=data_yaml, split='val')
    print(test_results.results_dict)
    wandb.log({f"test/{k}": v for k, v in test_results.results_dict.items()})

    wandb.finish()

if __name__ == "__main__":
    main()
