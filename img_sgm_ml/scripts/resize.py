import os

import cv2


# Resizes all the files in an directory

def resize(dir,
           shape,
           ending=None):  # "JPG" e.g.
    files = None
    hashes = []

    # shape[0] is width
    x = shape[0]  # If you want to resize the x site to a value and y proportional (e.g. 100)
    y = shape[1]  # If you want to resize the y site to a value and x proportional (e.g. 100)

    if ending is None:
        files = [f for f in os.listdir(dir) if not (f.startswith(".") or os.path.isdir(os.path.join(dir,f)))]
    else:
        files = [f for f in os.listdir(dir) if f.endswith(ending) and not (f.startswith(".") or os.path.isdir(os.path.join(dir,f)))]

    i = 0
    for f in files:
        f = os.path.join(dir, f)

        print(f'Working on {f}')

        img = cv2.imread(f)

        # Check for duplicates
        dhash = compute_hash(img)
        exists = None
        for h in hashes:
            if h["hash"] == dhash:
                exists = h["file"]
        if exists:
            print("-----DUPLICATE-----")
            print("Original: " + exists)
            print("Same: " + f)
            continue
        else:
            hashes.append({"hash": dhash, "file": f})

        if x is not None and y is not None:
            shape = shape
        elif x is not None:
            shape = (x, int(float(img.shape[0]) * (float(x) / float(img.shape[1]))))
        elif y is not None:
            shape = (int(float(img.shape[1]) * (float(y) / float(img.shape[0]))), y)

        print("Shape: ", shape, "Hash: ",dhash)

        img = cv2.resize(img, shape)
        cv2.imwrite(os.path.dirname(f) + "/resize/" + str(i) + ".jpg", img)
        i=i+1


def compute_hash(image, hash_size=8):
    # inspired by https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


if __name__ == "__main__":
    resize("/Users/tristanratz/Documents/03_Studium/8. Semester/BA/mat/Dataset/balloon/Unrelated", (700, None))
