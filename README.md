# Fourier feature PINN-based-multifrequency-multisource-Helmholtz-solver
**This repository reproduces the results of the paper "[Simulating seismic multifrequency wavefields with the Fourier feature physics-informed neural network.](https://academic.oup.com/gji/article/232/3/1503/6758508)" Geophysical Journal International 232, 1503–1514.

# Overview

We propose to use the Fourier feature physics Informed Neural Network (PINN) to solve for multifrequency multisource scattered wavefields for the Helmholtz equation. The proposed method breaks the limitation of the numerical solver in single-frequency wavefield simulation. The structure of the Fourier feature PINN is shown below.
![FFPINN-en](https://github.com/songc0a/Fourier-feature-PINN-based-multifrequency-multisource-Helmholtz-solver/assets/31889731/35539da5-41c8-4fa5-bfb0-23fd066a3cfc width="10px")

For the velocity in (a), the scattered wavefields (5-10 Hz) from the finite difference are shown in (b), and the scattered wavefields (5-10 Hz) from the  Fourier feature PINN are shown in (c).
![FFPINN](https://github.com/songc0a/Fourier-feature-PINN-based-multifrequency-multisource-Helmholtz-solver/assets/31889731/893e2f4b-58f1-4f5d-a4e9-6bfeba45e52a width="10px")

# Installation of Tensorflow1

CPU usage: pip install --pre "tensorflow==1.15.*"

GPU usage: pip install --pre "tensorflow-gpu==1.15.*"

# Code explanation

helm_solver_ffpinn_4D.py: Tensorflow code for solving the multifrequency-multisource scattered wavefields using Fourier feature PINN  
helm_solver_ffpinn_4D_test.py: Tensorflow code for a new velocity using the saved model
Sigsbee_sourceinput_data_generation_fre.m: Matlab code for generating training and test data  

# Citation information

If you find our codes and publications helpful, please kindly cite the following publications.

@article{song2023simulating,
  title={Simulating seismic multifrequency wavefields with the Fourier feature physics-informed neural network},
  author={Song, Chao and Wang, Yanghua},
  journal={Geophysical Journal International},
  volume={232},
  number={3},
  pages={1503--1514},
  year={2023},
  publisher={Oxford University Press}
}

# contact information
If there are any problems, please contact me through my emails: chao.song@kaust.edu.sa;chaosong@jlu.edu.cn
