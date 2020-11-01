# Image Segmentation Module for Label Studio

This module was build facilitate the creation of large image segmentation datasets with artificial intelligence with label-studio.
Create image segmentations and get in the fly predictions for the dataset your labeling.

It was build with the help of Matterports MaskRCNN implementation and is meant to help you label images.
After configuring the program for your needs and labeling a certain amount of images, the program should start to label images itself.

## Installation

First clone the project

```bash
git clone --recurse-submodules -j8 git://github.com/tristanratz/bar.git
```

Make sure you have docker and docker-compose installed to start the ml module.

### Run

To start the ML Module change to the ```img-sgm-ml``` folder and execute the following.
```bash
docker-compose up -d # The -d stands for no output
```
The ML-Image-Segmentation Application is now up and running. Open http://localhost:8080/

Now you have to label about 70 images. After you are done, change to the model tab in label studio and train the model.
After it finished learning you can continue labeling and should get predictions for your images.

Let the pre-labeling begin...!

## Own data

To create own labels go into ```img_sgm_ml/model/config.py``` and enter your labels in the following format

```python
```

After that add the images you want to label with the following command to the labeling tool:

```bash
label-studio init -i ./upload/ --input-format image-dir
```

or import them via the web interface (http://localhost:8080/import). Then (re)start the program.

## Disclaimer

This project was created for my Bachelor Thesis at the TU Darmstadt. (read it here)

A special thanks to the Finanz Informatik Solutions Plus GmbH for the great support!