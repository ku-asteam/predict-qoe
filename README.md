# Predict-QoE
Deeplearning-based QoE Prediction Model

## Introduction

Deeplearning-based QoE Prediction Model is a implementation of aSTEAM Project (Next-Generation Information Computing Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT). The function of this software is to predict the user satisfaction index (Quality of Experience) based on the CSI (Channel State Information) data collected when a user accesses a web page.

## Requirements and Dependencies
* Above development was based on the Python version of 3.5 (64bit)
* Please import packages (tensorflow, sklearn.decomposition.PCA, itertools, numpy, sklearn.preprocessing.MinMaxScaler, sklearn.metrics.mean_squared_error)

## Instructions
1. Get raw data (CSI, QoE)
2. Pre-processing such as cropping, clearing, and reducing dimensions
3. Creating and learning Deep Learning Prediction Model
4. Evaluation of Deep Learning Prediction Model
