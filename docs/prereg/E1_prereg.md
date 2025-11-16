# E1 – Baseline-Linear/short

Goal is: Establish a clean baseline for CIFAR-10 diffusion with linear β and NFE=50 FID.

Model: unet_cifar32

Data: CIFAR-10 train, batch_size=4, subset=None (full train); val = CIFAR-10 test.


Schedule:

β schedule: linear
Training: 10,000 optimizer steps.

Primary metric (for E1):
FID@NFE=50 on 10,000 generated samples (CIFAR-10 stats).

Measured once, at the end of training, with a fixed seed. 

Seed: 1077 for training; eval uses same internal seed routine.

Stop rule for E1: Stop at exactly 10,000 train steps. No early stopping, no hyperparam tuning.