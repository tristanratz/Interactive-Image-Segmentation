import os

import cv2

# Resizes all the files in an directory

if __name__ == "__main__":
    dir = "../../img_sgm/upload"
    ending = None  # "JPG" e.g.
    files = None

    # shape[0] is width
    shape = None  # Resize shape e.g. (100,100)
    x = 700  # If you want to resize the x site to a value and y proportional (e.g. 100)
    y = None  # If you want to resize the y site to a value and x proportional (e.g. 100)

    if ending is None:
        files = [f for f in os.listdir(dir)]
    else:
        files = [f for f in os.listdir(dir) if f.endswith('.json')]

    for f in files:
        f = os.path.join(dir, f)

        print(f'Working on {f}')

        img = cv2.imread(f)

        if x is not None:
            shape = (x, int(float(img.shape[0]) * (float(x) / float(img.shape[1]))))
        elif y is not None:
            shape = (int(float(img.shape[1]) * (float(y) / float(img.shape[0]))), y)
        print("Shape: ", shape)

        img = cv2.resize(img, shape)

        cv2.imwrite(f, img)
