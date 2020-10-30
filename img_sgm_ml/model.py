import sys
import os

from img_sgm_ml.rle.encode import encode
from img_sgm_ml.rle.decode import decode

MRCNN = os.path.abspath("./img_sgm_ml/Mask_RCNN/")
MRCNN2 = os.path.abspath("./Mask_RCNN/")
sys.path.append(MRCNN)
sys.path.append(MRCNN2)

from itertools import groupby
from label_studio.ml import LabelStudioMLBase
import mrcnn.model as modellib
from img_sgm_ml.train_mask_rcnn.config import LabelConfig
from label_studio.ml.utils import get_single_tag_keys
import matplotlib.pyplot as plt
import skimage
import numpy as np


class MaskRCNNModel(LabelStudioMLBase):

    model = None

    def __init__(self, **kwargs):
        super(MaskRCNNModel, self).__init__(**kwargs)

        if os.getenv("DOCKER"):
            self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
                  self.parsed_label_config, 'BrushLabels', 'Image')
        else:
            self.from_name = ""
            self.to_name = ""

        #weights_path = model.find_last()
        self.config = LabelConfig()

        # Create Model
        self.model = modellib.MaskRCNN("inference", self.config, "./checkpoints")

        # Load weights
        self.model.load_weights(os.path.join(os.path.abspath("./img_sgm_ml/rsc"), "mask_rcnn_coco.h5"), by_name=True)

    def predict(self, tasks, **kwargs):
        # Array with loaded images
        images = []

        for task in tasks:
            # Replace localhost with docker container name, when running with docker
            if os.getenv("DOCKER"):
                task['data']['image'] =  task['data']['image'].replace("localhost", "labeltool", 1)
            # Run model detection
            print("Running on {}".format(task['data']['image']))
            # Read image
            image = skimage.io.imread(task['data']['image'])
            images.append(image)

        # Detect objects
        predictions = self.model.detect(images, verbose=1)
        print("Inference finished. Start conversion.")

        # Build the detections into an array
        results = []
        for prediction in predictions:
            # labels = []
            # for id in prediction["class_ids"]:
            #     labels.append(self.config.CLASSES[id])

            for i in range(prediction['masks'].shape[2]):
                shape = np.array(prediction['masks'][:, :, i]).shape

                # Expand mask with 3 other dimensions
                mask_image = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
                mask_image[:, :, -1:] = np.expand_dims(prediction['masks'][:, :, i], axis=2)
                mask_image = mask_image*255

                if not os.getenv("DOCKER"):
                    plt.imsave(f"./out/mask_{i}.png", prediction['masks'][:, :, i])
                    plt.imsave(f"./out/mask_{i}_expanded.png",mask_image[:, :, 3])

                flat = mask_image.flatten() #swapaxes
                rle = encode(flat, len(flat))
                # rle = encode(np.reshape(prediction['masks'][:, :, i], shape[0]*shape[1]), shape[0]*shape[1])

                if not os.getenv("DOCKER"):
                    print("Encode and decode did work:", np.array_equal(flat, decode(rle)))
                    width = 700
                    height = 468
                    plt.imsave(f"./out/mask_{i}_flattened.png", np.reshape(flat, [height, width, 4]))

                results.append({
                    "result": [{
                        'from_name': self.from_name,
                        'to_name': self.to_name,
                        'type': 'brushlabels',
                        'value': {
                            'brushlabels': [self.config.CLASSES[prediction["class_ids"][i]]],
                            "format": "rle",
                            "rle": rle.tolist(),
                        },
                    }],
                    "score": float(prediction["scores"][i]),
                })

                # Convert to segmentation/polygon format
                # https://github.com/cocodataset/cocoapi/issues/131
        # print(results)
        return results

    def fit(self, completions, workdir=None, **kwargs):
        pass


if __name__ == "__main__":
    m = MaskRCNNModel()
    predictions = m.predict(tasks=[{
        "data": {
            # "image": "http://localhost:8080/data/upload/2ccf6fecb6406e9b3badb399f85070e3-DSC_0020.JPG"
            "image": "http://localhost:8080/data/upload/0462f5361cfcd2d02f94d44760b74f0c-DSC_0296.JPG"
        }
    }])

    #print(predictions)
    width = 700
    height = 468

    for p in predictions:
        plt.imsave(f"./out/{p['result']['value']['brushlabels'][0]}.png", np.reshape(decode(p["result"]["value"]["rle"]), [height, width, 4])[:,:,3]/255)
    print("Finished with code 0.")