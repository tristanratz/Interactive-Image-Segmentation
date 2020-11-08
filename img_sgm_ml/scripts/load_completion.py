import label_studio_converter.brush as b
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    b.convert_task_dir("../../img_sgm/completions", "./out")
    arr = np.load("./out/task-82.npy")
    plt.imsave("out/task82.png", arr)

# import sys
# import os
# sys.path.append(os.path.abspath("../rle/"))
# from encode import encode
#
# rle_own = encode(image, len(image))
#
# print(np.array_equal(rle, rle_own))
#
# for i, x in enumerate(rle):
#     if x != rle_own[i]:
#         print(i, x, rle_own[i])
