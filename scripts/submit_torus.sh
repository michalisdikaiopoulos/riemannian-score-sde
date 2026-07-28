#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J torus_k4
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u michalisdikaiopoulos@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o outputs/logs/gpu_torus_%J.out
#BSUB -e outputs/logs/gpu_torus_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

# Activate our project venv (instead of the conda "manifm" env)
cd /zhome/c2/5/213891/Thesis/riemannian-score-sde
source .venv/bin/activate

# Manifold: Torus (T^2), mixture of wrapped normals with K=4 components
python main.py \
    experiment=tn \
    n=2 \
    steps=50000 \
    seed=31 \
    beta_schedule.beta_f=15 \
    loss=ism \
    dataset.K=4 \
    val_freq=2000 \
    eval_batch_size=2048
