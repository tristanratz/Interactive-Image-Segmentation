from img_sgm_ml.rle.decode import decode
from img_sgm_ml.rle.encode import encode
import numpy as np


if __name__ == "__main__":
    x = np.array([123,12,33,123,2351,45,6,57,157,254,56,67], dtype=np.uint8)

    enc = encode(x, len(x))
    dec = decode(enc)

    print("Test worked.", np.array_equal(dec, x))
