# Predict-QoE
Deeplearning-based QoE Prediction Model

## Introduction

Deeplearning-based QoE Prediction Model is a implementation of aSTEAM Project (Next-Generation Information Computing Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT). The function of this software is to predict the user satisfaction index (Quality of Experience) based on the CSI (Channel State Information) data collected when a user accesses a web page.

## Requirements and Dependencies
* Above development was based on the Python version of 3.5 (64bit)
* Please import packages (tensorflow, sklearn.decomposition.PCA, itertools, numpy, sklearn.preprocessing.MinMaxScaler, sklearn.metrics.mean_squared_error)

## Instructions
* Import CSI, QoE data
  1. Get raw data
  2. Pre-processing such as clearing, and reducing dimensions
  3. Cropping whole data to training data and test data
* Create Deep Learning Prediction Model
  1. Create a prediction model consisting of 4 layers, a cost function, and an Adamoptimizer
  2. Learn the generated prediction model through training data
* Test Deep Learning Prediction Model
  1. Test Predection Model through test data
  2. Evaluate of Deep Learning Prediction Model
