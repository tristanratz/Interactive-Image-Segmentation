# Image Segmentation Module for Label Studio

This module was build facilitate the creation of large image segmentation datasets with the help of artificial intelligence with label-studio.
Create image segmentations and get on the fly predictions for the dataset your labeling.

It was build with the help of Matterports MaskRCNN implementation and is meant to help you label images.
After configuring the program for your needs and labeling a certain amount of images, the program should start to label images itself.

![Demo Image](https://github.com/tristanratz/Label-Studio-image-segmentation-ML-Module/blob/main/rsc/user_interface.png?raw=true)

## Installation

First clone the project

```bash
git clone --recurse-submodules -j8 https://github.com/tristanratz/Interactive-Image-Segmentation.git
```

Make sure you have docker and docker-compose installed to start the ml module.

## Run

To start the ML Module change to the ```img-sgm-ml``` folder and execute the following.
```bash
docker-compose up -d # The -d stands for no output
```
The ML-Image-Segmentation Application is now up and running. Open http://localhost:8080/

Now you have to label about 70 images. After you are done, change to the model tab in label studio and train the model.
After it finished learning you can continue labeling and should get predictions for your images.

Let the pre-labeling begin...!

## Use with own data

To create own labels go into ```img_sgm_ml/model/config.py``` and enter your labels in the following format

```python
CLASSES = {
                0: '__background__',
                1: 'YOUR_FIRST_CLASS',
                2: ...
          }       
```
Make sure that you safe the first spot for ```__background__```.
After that enter your class count (+1 for background)

Lastly add the images you want to label with the following command to the labeling tool:

```bash
label-studio init -i ./upload/ --input-format image-dir
```

or import them via the web interface (http://localhost:8080/import). Then (re)start the program.
Now you can label your images and train the model under http://localhost:8080/model.

#### Change of classes

Make sure you delete the contents of ```rsc/checkpoints``` directory after you decided to change the classes you train on. 
Otherwise the model will try to load an existing model and will try to train it even if the classes do not match.

## Use your own model

This projects holds a complete implementation to communicate with Label Studio and to predicting image segmentations with MaskRCNN.
If you want to use a diffrent model for image segmentation you may have to rewrite or change the ```img_sgm_ml/model/model.py``` and
```img_sgm_ml/model/dataset.py``` classes. 

This project contains code to transform the Label Studio format into the Matterport/MaskRCNN format.

## Dataset and pretrained model

Under ```rsc``` resides a dataset consisting out of ballons, it is devided into three subsets, which were used to test this tool.
The first dataset consists out of 70 labeled images of balloons, the second dataset are additional 70 images of balloons to test semi-automatic labeling, and the last set consists out of 10 unrelated images.
The model which resulted out of the testing can be found under ```rsc/checkpoints```.
It was trained 5 epochs on the images of the first dataset.

## Citation
```bibtex
@misc{LS_IMG_SGM_MODULE,
  title={{Interactive Labeling}: Facilitating the creation of comprehensive datasets with artificial intelligence},
  url={https://github.com/tristanratz/Label-Studio-image-segmentation-ML-Module},
  note={Open source software available from ttps://github.com/tristanratz/Label-Studio-image-segmentation-ML-Module},
  author={
    Tristan Ratz
  },
  year={2020},
}
```

## Disclaimer

This project was created for my Bachelor Thesis "Facilitating the creation of comprehensive datasets with
artificial intelligence" at the TU Darmstadt. (read it here)

A special thanks to the Finanz Informatik Solutions Plus GmbH for the great support!
