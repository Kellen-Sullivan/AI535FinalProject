from ultralytics import YOLO
# from ultralytics.data.augment import Albumentations, UnderwaterColorAttenuation, UnderwaterHaze
from IPython.display import Image
from roboflow import Roboflow
import wandb
import argparse
# from augmentations import UnderwaterColorAttenuation, UnderwaterHaze
import albumentations as A

rf = Roboflow(api_key="fZeOJWIPyFVGlTI0Q5lj")
project = rf.workspace("kellens-workspace-ausjh").project("underwater-trash-segmentation-io1hv")
version = project.version(3)
dataset = version.download("yolov11")

MODELS = [
    "yolo11s-seg.pt",
    "yolo11n-seg.pt"
]

def on_fit_epoch_end(trainer):
    epoch = trainer.epoch + 1
    wandb.log({"epoch": epoch, **trainer.metrics})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolo11n-seg.pt", help="The name of the YOLO model to use for training (e.g., 'yolo11n-seg.pt').")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model.")
    parser.add_argument("--optimizer", type=str, default="SGD", help="The optimizer to use for training (e.g., 'Adam', 'SGD').")
    parser.add_argument("--learning_rate_list", nargs="+", type=float, default=[0.01, 0.001, 0.0005], help="The learning rate to use for training.")
    parser.add_argument("--wandb_group", type=str, default="unnamed", help="The group name for the Weights & Biases run.")
    parser.add_argument("--wandb_name", type=str, default="unnamed", help="The name of the Weights & Biases run.")
    args = parser.parse_args()

    data_yaml = f"{dataset.location}/data.yaml"

    # Check if the specified model is in the list of available models
    if args.model_name not in MODELS:
        print(f"Error: Specified model '{args.model_name}' is not in the list of available models: {MODELS}")
        return

    for lr in args.learning_rate_list: 
        print(f"\nTraining with learning rate: {lr}")
        wandb.init(
            project="yolo11-trash-seg", 
            group=args.wandb_group, 
            name=f"{args.wandb_name}_lr_{lr}", 
            config={
                "epochs": args.epochs,
                "dataset": data_yaml,
                "model": args.model_name,
                "optimizer": args.optimizer,
                "learning_rate": lr
            }
        )

        # Load the seg model
        model = YOLO(args.model_name)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(task='segment', mode='train', data=data_yaml, epochs=args.epochs, optimizer=args.optimizer, lr0=lr)

        test_results = model.val(data=data_yaml, split='test')
        print(test_results.results_dict)
        wandb.log({f"test/{k}": v for k, v in test_results.results_dict.items()})

        wandb.finish()

if __name__ == "__main__":
    main()
