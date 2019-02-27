# deep_crs_him8
A deep learning approach to CRS model from Himawari8 data

This repository contains different models to reproduce the output of the CRS model developed by the Australia Bureau of Meteorology using raw reflectance data from Himawari 8 satellite.

A data set has been produced to test different approaches to infer CRS 1H precipitation. A data set to train models has been produced resulting in nearly 2500 samples of 400x400 patches containing the output of 1H CRS precipitation and the equivalent (7-16) reflectance bands from Himawari 8.

The following figure represents three samples of the data, being CRS output at the top and three Himawari bands (7, 10, 12) for the corresponding time and location.

<p align="center">
  <img src="CRS_Him8.png" width="300" title="CRS and Him8 bands">
</p>
