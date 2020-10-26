# Image Segmentation Module for Label Studio

This module is build for label studio to facilitate the creation of large image segmentation datasets with artificial intelligence.

You get on the fly predictions for the dataset your labeling.

## Installation

First clone the project

```bash
git clone --recurse-submodules -j8 git://github.com/tristanratz/bar.git
```

Then we have to install all dependencies.

```bash
pip install -r requirements.txt
```

Make sure you have docker and docker-compose installed to start the ml module.

Download the pretrained Matterport COCO Model from Matterport https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

### Run

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

## Own data

To create own labels go into ```img-sgm/config.xml``` and enter your labels in the following format

```xml
<Label value="LABEL_NAME" background="LABEL_COLOR"/>
```
e.g.
```xml
<Label value="Airplane" background="red"/>
```

After that add the images you want to label with the following command to the labeling tool:

```bash
label-studio init -i ./upload/ --input-format image-dir
```

or import them via the web interface.

## Disclaimer

This project was created for my Bachelor Thesis at the TU Darmstadt. (read it here)

A special thanks to the Finanz Informatik Solutions Plus GmbH for the great support!