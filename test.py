import cv2
import numpy as np
# write numpy arrays as images

w, h = 4, 4
data = np.zeros((h, w, 3))

data[1, 0:4] = [255, 0, 0]
data[2, 0:4] = [100, 100, 100]
data[3, 0:4] = [0, 100, 100]

nonlineardata = data / 255

cv2.namedWindow('write_window', cv2.WINDOW_AUTOSIZE)
cv2.imshow('write_window', data)
cv2.imwrite('RGB_eg.jpg', data) # don't write image as jpg
cv2.imwrite('RGB_eg.png', data)


invgamma = lambda x: x / 12.92 if x < 0.03928 else np.power((x + .055) / 1.055, 2.4)
invgammaV = np.vectorize(invgamma)

lineardata = invgammaV(nonlineardata)

xyz_trans = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])

xyzdata = np.zeros((h, w, 3))

for i in range(0, h):
    for j in range(0, w):
        xyzdata[i, j] = xyz_trans.dot(lineardata[i, j])

xyYdata = np.zeros((h, w, 3))

for i in range(0, h):
    for j in range(0, w):
        sum = np.sum(xyzdata[i, j])
        if sum == 0: continue
        xyYdata[i, j] = np.array([xyzdata[i, j][0] / sum, xyzdata[i, j][1] / sum, xyzdata[i, j][1]])

uw = (4 * 0.95) / (0.95 + 15 + 3 * 1.09)
vw = (9) / (0.95 + 15 + 3 * 1.09)

tldata = np.zeros((h, w, 3))
tldatastretched = np.zeros((h, w, 3))

for i in range(0, h):
    for j in range(0, w):
        t = xyzdata[i, j][0]
        l = 116 * np.power(t, 1 / 3) - 16 if t > 0.008856 else 903.3 * t
        d = xyzdata[i, j][0] + 15 * xyzdata[i, j][1] + 3 * xyzdata[i, j][2]
        if d == 0: continue
        udash = 4 * xyzdata[i, j][0] / d
        vdash = 9 * xyzdata[i, j][1] / d
        u = 13 * l * (udash - uw)
        v = 13 * l * (vdash - vw)
        tldata[i, j] = np.array([l, u, v])
        tldatastretched[i, j] = np.array([l + 10, u, v])

newxyzdata = np.zeros((h, w, 3))

for i in range(0, h):
    for j in range(0, w):
        udash = (tldatastretched[i, j][1] + 13 * uw * tldatastretched[i, j][0]) / 13 * tldatastretched[i, j][0]
        vdash = (tldatastretched[i, j][1] + 13 * vw * tldatastretched[i, j][0]) / 13 * tldatastretched[i, j][0]
        y = np.power(((tldatastretched[i, j][0] + 16) / 116),3) if tldatastretched[i, j][0] > 7.996 else tldatastretched[i, j][0] / 903.3
        if vdash == 0:
            newxyzdata[i, j] = np.array([0, y, 0])
        else:
            newxyzdata[i, j] = np.array([y * 2.25 * udash / vdash, y, y * ( 3 - 0.75 * udash - 5 * vdash ) / vdash])

rgb_trans = np.array([[3.240479,-1.53715,-0.498535], [-0.969256,1.875991,0.041556], [0.055648,-0.204043,1.057311]])

rgbdata = np.zeros((h, w, 3))

for i in range(0, h):
    for j in range(0, w):
        rgbdata[i, j] = rgb_trans.dot(newxyzdata[i, j])
        for k in range(0, 3):
            if rgbdata[i, j][k] < 0 : rgbdata[i, j][k] = 0
            if rgbdata[i, j][k] > 1: rgbdata[i, j][k] = 1
            rgbdata[i, j][k] = 12.92 * rgbdata[i, j][k] if rgbdata[i, j][k] < 0.00304 else (1.055 * np.power(rgbdata[i, j][k], 1/2.4) - 0.055)
            rgbdata[i, j][k] = rgbdata[i, j][k] * 255


a = 1
