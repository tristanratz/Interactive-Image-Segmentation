import os
import time

import mrcnn.model as modellib

from img_sgm_ml.model.utils import download_weights


class MaskRCNNModel():
    def __init__(self, config):
        self.config = config

        self.model_dir_path = os.path.abspath("./img_sgm_ml/model/checkpoints/")

        # Create Model
        self.model = modellib.MaskRCNN("inference", config, self.model_dir_path)

        # Download wights
        download_weights()

        if 0 < len([file for file in os.listdir(self.model_dir_path)
                    if not (file.endswith(".py") or file.startswith("."))]) and not os.getenv("COCO"):
            self.model_path = self.model.find_last()[0]
        else:
            self.model_path = os.path.join(os.path.abspath("./img_sgm_ml/rsc"), "mask_rcnn_coco.h5")

        # Load weights
        self.model.load_weights(self.model_path, by_name=True)

    def train(self, train_set, test_set):
        # https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d
        model = modellib.MaskRCNN(mode="training", config=self.config, model_dir='./checkpoints')
        model.load_weights('../rsc/mask_rcnn_coco.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

        model.train(train_set, test_set, learning_rate=2 * self.config.LEARNING_RATE, epochs=5, layers='heads') #layers=’heads’)
        # history = model.keras_model.history.history
        model_path = '../rsc/mask_rcnn_' + '.' + str(time.time()) + '.h5'
        model.keras_model.save_weights(model_path)

    def interference(self, img):
        return self.model.detect([img], verbose=1)

    def batchInterfere(self, imgs):
        return self.model.detect(imgs, verbose=1)
