import sys
import os

from img_sgm_ml.model.utils import generate_config, generate_color
from img_sgm_ml.rle.encode import encode
from img_sgm_ml.rle.decode import decode

MRCNN = os.path.abspath("./img_sgm_ml/Mask_RCNN/")
MRCNN2 = os.path.abspath("./Mask_RCNN/")
sys.path.append(MRCNN)
sys.path.append(MRCNN2)

from label_studio.ml import LabelStudioMLBase
from img_sgm_ml.model.config import LabelConfig
from label_studio.ml.utils import get_single_tag_keys
from img_sgm_ml.model.model import MaskRCNNModel
import matplotlib.pyplot as plt
import skimage
import numpy as np


class ModelAPI(LabelStudioMLBase):
    model = None

    def __init__(self, **kwargs):
        super(ModelAPI, self).__init__(**kwargs)

        if os.getenv("DOCKER"):
            self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
                self.parsed_label_config, 'BrushLabels', 'Image')
        else:
            self.from_name = ""
            self.to_name = ""

        # weights_path = model.find_last()
        self.config = LabelConfig()

        # Load model
        self.model = MaskRCNNModel(self.config)

        # Generate config
        generate_config(self.config, overwrite=False)

    def predict(self, tasks, **kwargs):
        """
        This method will call the prediction and will format the results into the format expected by label-studio

        Args:
            tasks: Array of tasks which will be predicted
            **kwargs:

        Returns: label-studio compatible results

        """
        # Array with loaded images
        images = []

        for task in tasks:
            # Replace localhost with docker container name, when running with docker
            if os.getenv("DOCKER"):
                task['data']['image'] = task['data']['image'].replace("localhost", "labeltool", 1)
            # Run model detection
            print("Running on {}".format(task['data']['image']))
            # Read image
            image = skimage.io.imread(task['data']['image'])
            images.append(image)

        # Detect objects
        predictions = self.model.batchInterfere(images)
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
                mask3d = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
                expanded_mask = np.array(prediction['masks'][:, :, i:i + 1])
                mask3d[:, :, :] = expanded_mask

                # Farbe laden
                (r, g, b, a) = generate_color(self.config.CLASSES[prediction["class_ids"][i]])

                mask_image = np.empty((shape[0], shape[1], 4), dtype=np.uint8)
                mask_image[:, :] = [b, g, r, a]

                # multiply mask_image with array to get a bgr image with the color of the mask/name
                mask_image = np.array(mask_image * mask3d, dtype=np.uint8)

                if not os.getenv("DOCKER"):
                    plt.imsave(f"./out/mask_{i}.png", prediction['masks'][:, :, i])
                    plt.imsave(f"./out/mask_{i}_expanded.png", mask3d[:, :, 3])
                    plt.imsave(f"./out/mask_{i}_color.png", mask_image)

                flat = mask_image.flatten()  # swapaxes
                rle = encode(flat, len(flat))
                # rle = encode(np.reshape(prediction['masks'][:, :, i], shape[0]*shape[1]), shape[0]*shape[1])

                if not os.getenv("DOCKER"):
                    print("Encode and decode did work:", np.array_equal(flat, decode(rle)))
                    plt.imsave(f"./out/mask_{i}_flattened.png", np.reshape(flat, [shape[1], shape[0], 4]))

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

    def fit(self, completions, workdir=None, **kwargs) -> None:
        """
        Formats the completions and images into the expected format

        Args:
            completions: Array of labeled images
            workdir:
            **kwargs:

        """
        pass


if __name__ == "__main__":
    m = ModelAPI()
    predictions = m.predict(tasks=[{
        "data": {
            # "image": "http://localhost:8080/data/upload/2ccf6fecb6406e9b3badb399f85070e3-DSC_0020.JPG"
            "image": "http://localhost:8080/data/upload/0462f5361cfcd2d02f94d44760b74f0c-DSC_0296.JPG"
        }
    }])

    # print(predictions)
    width = 700
    height = 468

    for p in predictions:
        plt.imsave(f"./out/{p['result'][0]['value']['brushlabels'][0]}.png",
                   np.reshape(decode(p["result"][0]["value"]["rle"]), [height, width, 4])[:, :, 3] / 255)
