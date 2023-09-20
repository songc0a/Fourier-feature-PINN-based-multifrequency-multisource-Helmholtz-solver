# PINN-based-multi-frequency-Helmholtz-solver
**This repository reproduces the results of the paper "[Simulating seismic multifrequency wavefields with the Fourier feature physics-informed neural network.](https://academic.oup.com/gji/article/232/3/1503/6758508)" Geophysical Journal International 232, 1503–1514.

# Overview

We propose to use the physics Informed Neural Network (PINN) to solve the scattered form of acoustic isotropic and anisotropic wave equations

PINN reduces the computational cost by avoid computing the inverse of the impedance matrix, which is suitable for anisotropic large models 

The resulting scattered wavefields are free of numerical dispersion artifacts

![du](https://user-images.githubusercontent.com/31889731/116671800-09454080-a9aa-11eb-8e73-d23e85e58639.jpg)


# Installation of Tensorflow1

CPU usage: pip install --pre "tensorflow==1.15.*"

GPU usage: pip install --pre "tensorflow-gpu==1.15.*"

# Code explanation

helm_pinn_solver_layermodel.py: Tensorflow code for solving the Helmholtz equation using PINN  
helm_pinn_solver_layermodel_sx.py: Tensorflow code for solving the Helmholtz equation for multiple sources using PINN  
helm_pinn_vti_layermodel.py: Tensorflow code for solving the Helmholtz equation in acoustic VTI media for using PINN  
Layer_training_data_generation*.m: Matlab code for generating training and test data  

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
