import os
from label_studio.ml.utils import is_skipped
from img_sgm_ml.Mask_RCNN.mrcnn.utils import Dataset
from img_sgm_ml.model.utils import completion_to_mrnn, transform_url


class LabelDataset(Dataset):

    def __init__(self, config):
        super(LabelDataset, self).__init__()

        self.config = config

        # Add classes
        for idx in config.CLASSES:
            self.add_class("img", idx, config.CLASSES[idx])

    def load_completions(self, completions):
        """ Prepare the data
        completions: The already labeled images
        subset: Subset to load: train or val
        """
        print('Collecting completions...')
        for completion in completions:
            if is_skipped(completion):
                continue

            image_url = transform_url(completion['data']["image"])

            bit_dict = completion_to_mrnn(completion['completions'][0], self.config)

            self.add_image(
                "img",
                image_id=completion['completions'][0]["id"],  # use file name as a unique image id
                path=image_url,
                width=bit_dict["width"], height=bit_dict["height"],
                class_ids=bit_dict["class_ids"],
                bitmask=bit_dict["bitmaps"])

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "img":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        return info["bitmask"], info["class_ids"]

