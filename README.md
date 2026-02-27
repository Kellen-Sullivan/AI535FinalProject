# [cite_start]Autonomous Underwater Trash Detection and Instance Segmentation [cite: 1, 8]
[cite_start]**AI 535 Final Project** [cite: 1]
[cite_start]**Authors:** Kellen Sullivan, Siya Sonpatki, Aiden Gabriel [cite: 2]

## Abstract
[cite_start]The influx of 14 million tons of plastic into oceans annually presents a critical global environmental challenge[cite: 4]. [cite_start]Approximately 100,000 marine animals die each year due to ocean debris entanglement[cite: 5]. [cite_start]The economic toll reaches $3 billion in lost potential tourism, and damages to marine shipping, fisheries, and aquacultures[cite: 6]. [cite_start]Because manually monitoring and removing ocean debris is expensive and difficult, this project proposes a solution for autonomous trash detection and removal[cite: 7, 8]. [cite_start]We aim to create robust computer vision for autonomous underwater vehicles (AUVs) to perform precise, pixel-level instance segmentation[cite: 9, 11]. [cite_start]This allows AUVs to precisely grab debris[cite: 11].

## Dataset Description
[cite_start]This project utilizes the **TrashCan 1.0** dataset[cite: 13]. [cite_start]The images are sourced from J-EDI (JAMSTEC E-Library of Deep-sea Images)[cite: 16]. [cite_start]JAMSTEC stands for Japan Agency of Marine Earth Science and Technology[cite: 17]. 
* [cite_start]**Composition:** The dataset features 7212 annotated images containing observations of trash, ROVs, and undersea flora & fauna[cite: 14]. 
* [cite_start]**Annotations:** Instance segmentation annotations are provided as a bitmap (pixel-level mask) indicating which pixels in the image contain each object[cite: 15].
* [cite_start]**Variants:** There are two versions of the dataset[cite: 18]:
    * [cite_start]*TrashCan-Material:* 16 different classes, defined by material of the trash[cite: 19].
    * [cite_start]*TrashCan-Instance:* 22 different classes, defined by the type of object of the trash[cite: 20].
* [cite_start]**Objective:** The purpose of the dataset is to develop efficient and accurate trash detection methods for onboard robot deployment[cite: 21].

## Methodology
[cite_start]Our technical approach relies on adapting state-of-the-art architectures[cite: 22].

### Network Architecture and Transfer Learning
[cite_start]We initialize a YOLOv11 model pre-trained on COCO instance segmentation[cite: 23]. [cite_start]Through transfer learning, we fine-tune the model on the TrashCan 1.0 dataset[cite: 24]. 

### Domain-Specific Data Augmentation
[cite_start]To ensure model robustness against underwater environments, we utilize several data augmentation techniques[cite: 28]:
* [cite_start]**Geometric Transformations:** We apply horizontal flips, as well as cropping and resizing[cite: 29, 30].
* [cite_start]**Color Specific Attenuation:** Water absorbs red light the fastest, followed by green, and then blue, which causes red light to die off as distance increases[cite: 33]. [cite_start]To account for this, we use color specific attenuation to reduce red values the most, and reduce green values some[cite: 31, 32]. 
* [cite_start]**Global Haze Injection:** We add global haze to images by selecting a range of blue-green veiling colors, likely sampled from the dataset[cite: 34, 35]. [cite_start]For each pixel, we randomly select a strength of veiling color and blend the original color with this color[cite: 36].

## Evaluation
[cite_start]We evaluate performance using Mean Average Precision (mAP) for both bounding boxes and segmentation masks[cite: 25].

## References
* [cite_start][1] Condor Ferries, “100+ Ocean Pollution Statistics & Facts (2019),” Condor Ferries, 2019. https://www.condorferries.co.uk/marine-ocean-pollution-statistics-facts [cite: 40]
* [cite_start][2] “Plastic’s Hidden Price Tag: U.S. Faces Up to $1.1 Trillion in Annual Social Costs, Duke Scholars Estimate,” Duke.edu, 2025. https://nicholasinstitute.duke.edu/articles/plastics-hidden-price-tag-us-faces-11-trillion-annual-social-costs-duke-scholars-estimate [cite: 41]
* [3] B. C. Corrigan, Z. Y. Tay, and D. Konovessis, "Real-time instance segmentation for detection of underwater litter as a plastic source," J. Mar. Sci. Eng., vol. 11, no. 8, Art. no. [cite_start]1532, Aug. 2023. [cite: 42, 43]
* [4] Hong, Jungseok, Michael S. Fulton, and Junaed Sattar. 2020. TrashCan 1.0: An Instance-Segmentation Labeled Dataset of Trash Observations. Dataset. [cite_start]Data Repository for the University of Minnesota. [cite: 44, 45]
* [cite_start][5] Hong, J., M. Fulton, and J. Sattar, “TrashCan: A semantically-segmented dataset towards visual detection of marine debris,” arXiv preprint arXiv:2007.08097, 2020. [cite: 46]
* [6] R. Sapkota et al., "YOLO advances to its genesis: A decadal and comprehensive review of the You Only Look Once (YOLO) series," Artif. Intell. Rev., vol. [cite_start]58, 2025. [cite: 47, 48]
* [7] T.-Y. [cite_start]Lin et al., "Microsoft COCO: Common Objects in Context," arXiv preprint arXiv:1405.0312, 2015. [cite: 49]
