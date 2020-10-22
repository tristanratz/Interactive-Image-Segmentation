# Image Segmentation Module for Label Studio

This module is build for label studio to facilitate the creation of large image segmentation datasets with artificial intelligence.

You get on the fly predictions for the dataset your labeling.

## Installation

First we have to install label studio.

```bash
pip install label-studio
```

Make sure you have docker and docker-compose installed to start the ml module.

## Own data

To create own labels for the different 

## Start

To start the ML Module change to the ```img-sgm-ml``` folder and execute the following.
```bash
docker-compose up -d # The -d is for no output
```
The ML-Image-Segmentation Server is now up and running.

Now start label-studio to get started with labeling.
```bash
label-studio start img_sgm
```

Now you have to label about 70 images. After you are done, change to the model tab in label studio and train the model.
After it finished learning you can continue labeling and should get predictions for your images.

Let the pre-labeling begin...!

## Disclaimer

This project was created for my Bachelor Thesis at the TU Darmstadt. (read it here)

A special thanks to the Finanz Informatik Solutions Plus GmbH for the great support!