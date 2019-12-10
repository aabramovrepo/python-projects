#
# Alexey Abramov <alexey.abramov.salzi@gmail.com>
#
# Adaptive thresholding using the integral image
#

import cv2
import scipy
import numpy as np
from scipy import misc


def main():
    fname = 'images/test_1.png'

    im = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # cv2.imwrite('./output/raw.png', im)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BAYER_GB2GRAY)
    cv2.imwrite('output/gray.png', im_gray)
    im_arr_gray = np.asarray(im_gray, dtype=float)

    height = im_arr_gray.shape[0]
    width = im_arr_gray.shape[1]

    # build the integral image as a combination
    # of cumulative sums along both axes
    integral_img = np.cumsum(np.cumsum(im_arr_gray, axis=0), axis=1)
    # scipy.misc.imsave('output/integral_image.png', integral_img)

    # perform adaptive thresholding
    binary_img_adapt = np.zeros(im_arr_gray.shape)

    s = 30  # window size
    ratio = 0.88  # percentage used for thresholding

    for i in range(height):
        for j in range(width):

            # get coordinates of the window
            y1 = i - s / 2
            y2 = i + s / 2

            x1 = j - s / 2
            x2 = j + s / 2

            # check boundaries
            y1 = 1 if y1 < 1 else y1
            x1 = 1 if x1 < 1 else x1

            y2 = height - 1 if y2 >= height else y2
            x2 = width - 1 if x2 >= width else x2

            cnt = (x2 - x1) * (y2 - y1)
            val = integral_img[y2, x2] - integral_img[y1 - 1, x2] - \
                  integral_img[y2, x1 - 1] + integral_img[y1 - 1, x1 - 1]

            if im_arr_gray[i, j] * cnt < val * ratio:
                binary_img_adapt[i, j] = 0  # background
            else:
                binary_img_adapt[i, j] = 255  # foreground

    scipy.misc.imsave('output/adaptive_threshold.png', binary_img_adapt)

    # perform global thresholding for comparison
    binary_img_glob = np.zeros(im_arr_gray.shape)

    for i in range(height):
        for j in range(width):

            if im_arr_gray[i, j] < 55.:
                binary_img_glob[i, j] = 0  # background
            else:
                binary_img_glob[i, j] = 255  # foreground

    scipy.misc.imsave('output/global_threshold.png', binary_img_glob)


if __name__ == "__main__":
    main()
