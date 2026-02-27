# Autonomous Underwater Trash Detection and Instance Segmentation
**AI 535 Final Project**
**Authors:** Kellen Sullivan, Siya Sonpatki, Aiden Gabriel

## Notes
I made the dataset applicable for Yolo using Roboflow. Here is the link https://app.roboflow.com/kellens-workspace-ausjh/underwater-trash-segmentation-io1hv/2

## Abstract
The influx of 14 million tons of plastic into oceans annually presents a critical global environmental challenge. Approximately 100,000 marine animals die each year due to ocean debris entanglement. The economic toll reaches $3 billion in lost potential tourism, and damages to marine shipping, fisheries, and aquacultures. Because manually monitoring and removing ocean debris is expensive and difficult, this project proposes a solution for autonomous trash detection and removal. We aim to create robust computer vision for autonomous underwater vehicles (AUVs) to perform precise, pixel-level instance segmentation. This allows AUVs to precisely grab debris.

## Dataset Description
This project utilizes the **TrashCan 1.0** dataset. The images are sourced from J-EDI (JAMSTEC E-Library of Deep-sea Images). JAMSTEC stands for Japan Agency of Marine Earth Science and Technology. 
* **Composition:** The dataset features 7212 annotated images containing observations of trash, ROVs, and undersea flora & fauna. 
* **Annotations:** Instance segmentation annotations are provided as a bitmap (pixel-level mask) indicating which pixels in the image contain each object.
* **Variants:** There are two versions of the dataset:
    * *TrashCan-Material:* 16 different classes, defined by material of the trash.
    * *TrashCan-Instance:* 22 different classes, defined by the type of object of the trash.
* **Objective:** The purpose of the dataset is to develop efficient and accurate trash detection methods for onboard robot deployment.

## Methodology
Our technical approach relies on adapting state-of-the-art architectures.

### Network Architecture and Transfer Learning
We initialize a YOLOv11 model pre-trained on COCO instance segmentation. Through transfer learning, we fine-tune the model on the TrashCan 1.0 dataset. 

### Domain-Specific Data Augmentation
To ensure model robustness against underwater environments, we utilize several data augmentation techniques:
* **Geometric Transformations:** We apply horizontal flips, as well as cropping and resizing.
* **Color Specific Attenuation:** Water absorbs red light the fastest, followed by green, and then blue, which causes red light to die off as distance increases. To account for this, we use color specific attenuation to reduce red values the most, and reduce green values some. 
* **Global Haze Injection:** We add global haze to images by selecting a range of blue-green veiling colors, likely sampled from the dataset. For each pixel, we randomly select a strength of veiling color and blend the original color with this color.

## Evaluation
We evaluate performance using Mean Average Precision (mAP) for both bounding boxes and segmentation masks.
