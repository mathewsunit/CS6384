import cv2
import numpy as np
import sys

if (len(sys.argv) != 3):
    print(sys.argv[0], "takes 2 arguments. Not ", len(sys.argv) - 1)
    sys.exit()

name_input = sys.argv[1]
name_output = sys.argv[2]

image_input = cv2.imread(name_input, cv2.IMREAD_UNCHANGED);
if (image_input is None):
    print(sys.argv[0], "Failed to read image from ", name_input)
    sys.exit()
cv2.imshow('original image', image_input);

primary_image = image_input

cv2.imshow('color image', primary_image);
rows, cols, bands = primary_image.shape
green_image_output = np.zeros([rows, cols], dtype=np.uint8)

# this is slow but we are not concerned with speed here
for i in range(0, rows):
    for j in range(0, cols):
        green_image_output[i, j] = primary_image[i, j][1]

cv2.imshow('green output image', green_image_output);
cv2.imwrite(name_output, green_image_output);

max_image_output = np.zeros([rows, cols], dtype=np.uint8)

# this is slow but we are not concerned with speed here
for i in range(0, rows):
    for j in range(0, cols):
        max_image_output[i, j] = np.max(primary_image[i, j])

cv2.imshow('max output image', max_image_output);
cv2.imwrite(name_output, max_image_output);

round_avg_image_output = np.zeros([rows, cols], dtype=np.uint8)

# this is slow but we are not concerned with speed here
for i in range(0, rows):
    for j in range(0, cols):
        round_avg_image_output[i, j] = np.average(primary_image[i, j])

cv2.imshow('avg output image', round_avg_image_output);
cv2.imwrite(name_output, round_avg_image_output);

# wait for key to exit

round_weighted_avg_image_output = np.zeros([rows, cols], dtype=np.uint8)

# this is slow but we are not concerned with speed here
for i in range(0, rows):
    for j in range(0, cols):
        round_weighted_avg_image_output[i, j] = .3 * primary_image[i, j][0] + .6 * primary_image[i, j][1] + .1 * \
                                                primary_image[i, j][2]

cv2.imshow('weighted avg output image', round_weighted_avg_image_output);
cv2.imwrite(name_output, round_weighted_avg_image_output);

cv2.waitKey(0)
cv2.destroyAllWindows()
