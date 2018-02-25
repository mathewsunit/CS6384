import cv2
import numpy as np
import sys
# write numpy arrays as images

xyz_trans = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])
uw = (4 * 0.95) / (0.95 + 15 + 3 * 1.09)
vw = (9) / (0.95 + 15 + 3 * 1.09)
h = 0
w = 0

def rgb2nonlinearrgb(data):
    nonlineardata = data / 255
    return nonlineardata

def invgammacorrection(data):
    invgamma = lambda x: x / 12.92 if x < 0.03928 else np.power((x + .055) / 1.055, 2.4)
    invgammaV = np.vectorize(invgamma)
    lineardata = invgammaV(data)
    return lineardata

def lineartransform(data):
    xyzdata = np.zeros((h, w, 3))
    for i in range(0, h):
        for j in range(0, w):
            xyzdata[i, j] = xyz_trans.dot(data[i, j])
    return xyzdata

def computexyY(data):
    xyYdata = np.zeros((h, w, 3))
    for i in range(0, h):
        for j in range(0, w):
            sum = np.sum(data[i, j])
            if sum == 0: continue
            xyYdata[i, j] = np.array([data[i, j][0] / sum, data[i, j][1] / sum, data[i, j][1]])
    return xyYdata

def xyz2luv(data):
    tldata = np.zeros((h, w, 3))
    for i in range(0, h):
        for j in range(0, w):
            t = data[i, j][0]
            l = 116 * np.power(t, 1 / 3) - 16 if t > 0.008856 else 903.3 * t
            d = data[i, j][0] + 15 * data[i, j][1] + 3 * data[i, j][2]
            if d == 0: continue
            udash = 4 * data[i, j][0] / d
            vdash = 9 * data[i, j][1] / d
            u = 13 * l * (udash - uw)
            v = 13 * l * (vdash - vw)
            tldata[i, j] = np.array([l, u, v])
    return tldata

def luv2xyz(data):
    newxyzdata = np.zeros((h, w, 3))
    for i in range(0, h):
        for j in range(0, w):
            udash = (data[i, j][1] + 13 * uw * data[i, j][0]) / 13 * data[i, j][0]
            vdash = (data[i, j][1] + 13 * vw * data[i, j][0]) / 13 * data[i, j][0]
            y = np.power(((data[i, j][0] + 16) / 116),3) if data[i, j][0] > 7.996 else data[i, j][0] / 903.3
            if vdash == 0:
                newxyzdata[i, j] = np.array([0, y, 0])
            else:
                newxyzdata[i, j] = np.array([y * 2.25 * udash / vdash, y, y * ( 3 - 0.75 * udash - 5 * vdash ) / vdash])
    return newxyzdata

def xyz2rgb(data):
    rgb_trans = np.array([[3.240479,-1.53715,-0.498535], [-0.969256,1.875991,0.041556], [0.055648,-0.204043,1.057311]])
    rgbdata = np.zeros((h, w, 3))
    for i in range(0, h):
        for j in range(0, w):
            rgbdata[i, j] = rgb_trans.dot(data[i, j])
            for k in range(0, 3):
                if rgbdata[i, j][k] < 0 : rgbdata[i, j][k] = 0
                if rgbdata[i, j][k] > 1: rgbdata[i, j][k] = 1
                rgbdata[i, j][k] = 12.92 * rgbdata[i, j][k] if rgbdata[i, j][k] < 0.00304 else (1.055 * np.power(rgbdata[i, j][k], 1/2.4) - 0.055)
                rgbdata[i, j][k] = rgbdata[i, j][k] * 255
    return rgbdata

def main(args=None):
    """The main routine."""
    if (len(sys.argv) != 7):
        print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv) - 1)
        print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
        print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
        sys.exit()

    w1 = float(sys.argv[1])
    h1 = float(sys.argv[2])
    w2 = float(sys.argv[3])
    h2 = float(sys.argv[4])
    name_input = sys.argv[5]
    name_output = sys.argv[6]

    if (w1 < 0 or h1 < 0 or w2 <= w1 or h2 <= h1 or w2 > 1 or h2 > 1):
        print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
        sys.exit()

    inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
    if (inputImage is None):
        print(sys.argv[0], ": Failed to read image from: ", name_input)
        sys.exit()

    cv2.imshow("input image: " + name_input, inputImage)

    rows, cols, bands = inputImage.shape  # bands == 3
    W1 = round(w1 * (cols - 1))
    H1 = round(h1 * (rows - 1))
    W2 = round(w2 * (cols - 1))
    H2 = round(h2 * (rows - 1))

    # The transformation should be based on the
    # historgram of the pixels in the W1,W2,H1,H2 range.
    # The following code goes over these pixels

    tmp = np.copy(inputImage)

    for i in range(H1, H2):
        for j in range(W1, W2):
            b, g, r = inputImage[i, j]
            gray = round(0.3 * r + 0.6 * g + 0.1 * b + 0.5)
            tmp[i, j] = [gray, gray, gray]

    cv2.imshow('tmp', tmp)

    # end of example of going over window

    outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            b, g, r = inputImage[i, j]
            outputImage[i, j] = [b, g, r]
    cv2.imshow("output:", outputImage)
    cv2.imwrite(name_output, outputImage);

    # wait for key to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Do argument parsing here (eg. with argparse) and anything else
    # you want your project to do.

if __name__ == "__main__":
    main()