import sys
import os
MRCNN = os.path.abspath("./img_sgm_ml/Mask_RCNN/")
MRCNN2 = os.path.abspath("./Mask_RCNN/")
sys.path.append(MRCNN)
sys.path.append(MRCNN2)

from itertools import groupby
from label_studio.ml import LabelStudioMLBase
import mrcnn.model as modellib
from img_sgm_ml.train_mask_rcnn.config import LabelConfig
from label_studio.ml.utils import get_single_tag_keys
import skimage


class MaskRCNNModel(LabelStudioMLBase):

    model = None

    def __init__(self, **kwargs):
        super(MaskRCNNModel, self).__init__(**kwargs)

        #self.from_name = ""
        #self.to_name = ""
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
             self.parsed_label_config, 'BrushLabels', 'Image')

        #weights_path = model.find_last()
        self.config = LabelConfig()

        # Create Model
        self.model = modellib.MaskRCNN("inference", self.config, "./checkpoints")

        # Load weights
        self.model.load_weights(os.path.join(os.path.abspath("./img_sgm_ml/rsc"), "mask_rcnn_coco.h5"), by_name=True)

        # self.dataset =

    def predict(self, tasks, **kwargs):
        # Array with loaded images
        images = []
        for task in tasks:
            # Run model detection and generate the color splash effect
            print("Running on {}".format(task['data']['image']))
            # Read image
            image = skimage.io.imread(task['data']['image'])
            images.append(image)

        # Detect objects
        predictions = self.model.detect(images, verbose=1)

        # Build the detections into an array
        results = []
        for prediction in predictions:
            # labels = []
            # for id in prediction["class_ids"]:
            #     labels.append(self.config.CLASSES[id])

            results.append({
                "result": {
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'value': {
                        #'brushlabels': prediction["class_ids"],
                        'brushlabels': [self.config.CLASSES[id] for id in prediction["class_ids"]],
                        "format": "rle",
                        "rle": [self.binary_mask_to_rle(bm) for bm in prediction["masks"]]
                    }
                },
                "score": float(prediction["scores"])
            })

            # Convert to segmentation/polygon format
            # https://github.com/cocodataset/cocoapi/issues/131

        return results

    def binary_mask_to_rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle

    def fit(self, completions, workdir=None, **kwargs):
        pass


if __name__ == "__main__":
    m = MaskRCNNModel()
    print(m.predict(tasks=[{
        "data": {
            "image": "http://localhost:8080/data/upload/2ccf6fecb6406e9b3badb399f85070e3-DSC_0020.JPG"
        }
    }]))