Clusters on OSU HPC:

Partitions (-p):
    dgxh: H100
        - fastest GPUs, hardest to get
    ampere: A40
        - slower GPUs, very easy to get
    dgx2: V100
        - slower GPUs, can be hard to get
* Our code seems to be mostly limited by dataloading so the better GPUs don't seem to help much

sbatch - command to run jobs asyncronously
    - in train.sh all this lines at the top that start with SBATCH are settings for the job
    - to run train.sh with sbatch you'll have to uncomment 3 lines to load cuda and start the conda env
        - you'll also have to change two of the lines to fit you environment
    Terminal Command:
        sbatch ./train.sh


