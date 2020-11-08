import os
import time

import mrcnn.model as modellib

from img_sgm_ml.model.utils import download_weights


class MaskRCNNModel():
    def __init__(self, config):
        self.config = config

        self.model_dir_path = os.path.abspath("./img_sgm_ml/rsc/checkpoints/")

        # Create Model
        self.model = modellib.MaskRCNN("inference", config, self.model_dir_path)
        self.train_model = modellib.MaskRCNN("training", config, self.model_dir_path)
        self.model_path = ""

        # Download wights
        download_weights()

    def train(self, train_set, test_set):
        # https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d

        # Load latest models
        self.reset_model(train=True)
        model = self.train_model

        # Do training
        model.train(train_set, test_set, learning_rate=2 * self.config.LEARNING_RATE, epochs=5, layers='heads')
        # history = model.keras_model.history.history
        model_path = '../rsc/mask_rcnn_' + '.' + str(time.time()) + '.h5'
        model.keras_model.save_weights(model_path)
        return model_path

    def interference(self, img):
        # Load latest models and do interference when successfull
        if self.reset_model():
            return self.model.detect([img], verbose=1)
        else:
            return []

    def batchInterfere(self, imgs):
        # Load latest models and do interference when successfull
        if self.reset_model():
            return self.model.detect(imgs, verbose=1)
        else:
            return []

    def reset_model(self, train=False):
        coco_path = os.path.join(os.path.abspath("./img_sgm_ml/rsc"), "mask_rcnn_coco.h5")
        model_path = ""
        if 0 < len([file for file in os.listdir(self.model_dir_path)
                    if not (file.endswith(".py") or file.startswith(".") or file.startswith("label"))]) and not os.getenv("COCO"):
            model_path = self.model.find_last()[0]
            if train:
                self.train_model.load_weights(self.model_path, by_name=True)
                return True
        else:
            model_path = coco_path
            if train:
                self.train_model.load_weights(coco_path,
                                              by_name=True,
                                              exclude=["mrcnn_class_logits",
                                                       "mrcnn_bbox_fc",
                                                       "mrcnn_bbox",
                                                       "mrcnn_mask"])
                return True

        # if model_path is coco_path and not os.getenv("USECOCO"):
        #    return False

        if not self.model_path is model_path:
            # Load weights
            self.model.load_weights(model_path, by_name=True)
            self.model_path = model_path

        return True
