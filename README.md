# Fourier feature PINN-based-multifrequency-multisource-Helmholtz-solver
**This repository reproduces the results of the paper "[Simulating seismic multifrequency wavefields with the Fourier feature physics-informed neural network.](https://academic.oup.com/gji/article/232/3/1503/6758508)" Geophysical Journal International 232, 1503–1514.

# Overview

We propose to use the Fourier feature physics Informed Neural Network (PINN) to solve for multifrequency multisource scattered wavefields for the Helmholtz equation. The proposed method breaks the limitation of the numerical solver in single-frequency wavefield simulation.
![FFPINN-en](https://github.com/songc0a/Fourier-feature-PINN-based-multifrequency-multisource-Helmholtz-solver/assets/31889731/066826f7-c0ae-4188-bf29-8956fcd693ca)

![FFPINN](https://github.com/songc0a/Fourier-feature-PINN-based-multifrequency-multisource-Helmholtz-solver/assets/31889731/06bda160-ad30-499f-825a-5712b6ae5c83)


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
