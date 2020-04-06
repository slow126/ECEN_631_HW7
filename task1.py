import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

PATH = 'Time_To_Impact_Images'
files = os.listdir(PATH)
files.sort()
m = 1
SCALE = 1
PT_THRESHOLD = 20
LEVEL = 2
expansion = []
new_expan = []

for j in range(len(files) - m):
    file = files[j]
    frame = cv2.imread(os.path.join(PATH, file))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    center = [gray.shape[0] / 2, gray.shape[1] / 2]
    gray = gray[0:240, 250:400]

    corners = cv2.goodFeaturesToTrack(gray, 1000, .1, 1)

    next_file = files[j + m]
    next_frame = cv2.imread(os.path.join(PATH, next_file))
    next_frame = cv2.resize(next_frame, (int(next_frame.shape[1] / SCALE), int(next_frame.shape[0] / SCALE)))
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    next_gray = next_gray[0:240, 250:400]
    next_corners, status, err = cv2.calcOpticalFlowPyrLK(gray, next_gray, corners, None, maxLevel=LEVEL)

    goodPts = cv2.threshold(err, PT_THRESHOLD, 1, cv2.THRESH_BINARY_INV)

    corners = np.squeeze(corners)
    next_corners = np.squeeze(next_corners)
    retval, mask = cv2.findFundamentalMat(np.array(corners), np.array(next_corners), cv2.FM_RANSAC, 3, 0.9)

    if mask is not None:
        corners = corners * mask
        next_corners = next_corners * mask

    corners = corners + (240, 0)
    next_corners = next_corners + (240, 0)

    a = []
    new_a = []
    print(np.sum(goodPts[1]))
    for i in range(len(corners)):
        if (goodPts[1][i] != 0):
            if (tuple(next_corners[i]) != (0, 0) and tuple(corners[i]) != (0, 0)):
                cv2.circle(frame, tuple((np.squeeze(corners[i])).astype(int)), 1, (0, 255, 0), -1)
                cv2.line(frame, tuple((np.squeeze(corners[i])).astype(int)), tuple((np.squeeze(next_corners[i])).astype(int)), (255, 0, 255),
                         thickness=1)
                a.append(math.sqrt((corners[i][0] - next_corners[i][0])**2 + (corners[i][1] - next_corners[i][1])**2))
                dist2 = math.sqrt((next_corners[i][0] - center[1])**2 + (next_corners[i][1] - center[0])**2)
                dist1 = math.sqrt((corners[i][0] - center[1])**2 + (corners[i][1] - center[0])**2)
                new_a.append(dist2/dist1)


    expansion.append(np.mean(a))
    new_expan.append(np.mean(new_a))
    cv2.imshow("frame", frame)
    # cv2.imwrite('LK/' + SAVE_PATH + '/frame-' + str(j).zfill(5) + '.jpg', frame)
    cv2.waitKey(1)

expansion = new_expan
frame_num = np.arange(1, 18)
expansion = np.array(expansion)
tau = (expansion) / (expansion - 1)
plt.scatter(frame_num, tau)

out = np.polyfit(frame_num, tau, 1)
frame_num = np.arange(0, 40)
plt.plot(frame_num, out[0] * frame_num + out[1], color='red')
plt.plot(frame_num, np.zeros_like(frame_num))

plt.show()

