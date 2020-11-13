from img_sgm_ml.Mask_RCNN.mrcnn.config import Config


class LabelConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "label"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Learning rate
    LEARNING_RATE = 0.006

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Classes (including background)
    CLASSES = {
               0 : u'__background__',
               1 : u'backpack',
               2 : u'bench',
               3 : u'bottle',
               4 : u'Brechstange',
               5 : u'cell phone',
               6 : u'chair',
               7 : u'clock',
               8 : u'cup',
               9 : u'dining table',
               10: u'Flexschlauch',
               11: u'GasCombo',
               12: u'Gasflasche',
               13: u'Hammer',
               14: u'handbag',
               15: u'HEMA',
               16: u'keyboard',
               17: u'laptop',
               18: u'person',
               19: u'potted plant',
               20: u'Stirnlampe',
               21: u'Sturmmaske',
               22: u'suitcase',
               23: u'Tasche',
               24: u'tv',
               25: u'umbrella',
               26: u'vase',
               27: u'wine glass',
               28: u'Muetze'
               }

    # Coco classes (including background)
    # CLASSES = {0: u'__background__',
    #      1: u'person',
    #      2: u'bicycle',
    #      3: u'car',
    #      4: u'motorcycle',
    #      5: u'airplane',
    #      6: u'bus',
    #      7: u'train',
    #      8: u'truck',
    #      9: u'boat',
    #      10: u'traffic light',
    #      11: u'fire hydrant',
    #      12: u'stop sign',
    #      13: u'parking meter',
    #      14: u'bench',
    #      15: u'bird',
    #      16: u'cat',
    #      17: u'dog',
    #      18: u'horse',
    #      19: u'sheep',
    #      20: u'cow',
    #      21: u'elephant',
    #      22: u'bear',
    #      23: u'zebra',
    #      24: u'giraffe',
    #      25: u'backpack',
    #      26: u'umbrella',
    #      27: u'handbag',
    #      28: u'tie',
    #      29: u'suitcase',
    #      30: u'frisbee',
    #      31: u'skis',
    #      32: u'snowboard',
    #      33: u'sports ball',
    #      34: u'kite',
    #      35: u'baseball bat',
    #      36: u'baseball glove',
    #      37: u'skateboard',
    #      38: u'surfboard',
    #      39: u'tennis racket',
    #      40: u'bottle',
    #      41: u'wine glass',
    #      42: u'cup',
    #      43: u'fork',
    #      44: u'knife',
    #      45: u'spoon',
    #      46: u'bowl',
    #      47: u'banana',
    #      48: u'apple',
    #      49: u'sandwich',
    #      50: u'orange',
    #      51: u'broccoli',
    #      52: u'carrot',
    #      53: u'hot dog',
    #      54: u'pizza',
    #      55: u'donut',
    #      56: u'cake',
    #      57: u'chair',
    #      58: u'couch',
    #      59: u'potted plant',
    #      60: u'bed',
    #      61: u'dining table',
    #      62: u'toilet',
    #      63: u'tv',
    #      64: u'laptop',
    #      65: u'mouse',
    #      66: u'remote',
    #      67: u'keyboard',
    #      68: u'cell phone',
    #      69: u'microwave',
    #      70: u'oven',
    #      71: u'toaster',
    #      72: u'sink',
    #      73: u'refrigerator',
    #      74: u'book',
    #      75: u'clock',
    #      76: u'vase',
    #      77: u'scissors',
    #      78: u'teddy bear',
    #      79: u'hair drier',
    #      80: u'toothbrush'}

    # Number of classes (including background)
    NUM_CLASSES = 4 + 1  # Background + classes
