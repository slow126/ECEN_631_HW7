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
template = cv2.imread("template.jpg")

minHessian = 400
surf = cv2.xfeatures2d.SURF_create(minHessian)
flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
fx = 825.0900600547
can_width = 59


def my_resize(img, scale):
    small_img = np.zeros((int(img.shape[0] / scale), int(img.shape[1] / scale), img.shape[2]))
    for i in range(img.shape[2]):
        temp = img[:, :, i]
        small_img[:, :, i] = cv2.resize(temp, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    return small_img.astype("uint8")


def my_Homography2(template, scene, template_key, template_descript):
    # template_key, template_descript = surf.detectAndCompute(template, None)
    scene_key, scene_descript = surf.detectAndCompute(scene, None)

    # template_key, template_key = orb.detectAndCompute(template, None)
    # scene_key, scene_descript = orb.detectAndCompute(scene, None)

    matches = flann.knnMatch(template_descript, scene_descript, 2)

    my_thresh = 0.75
    good_matches = []
    for m, n in matches:
        if (m.distance < my_thresh * n.distance):
            good_matches.append(m)

    obj = []
    scene_pts = []
    for i in range(len(good_matches)):
        temp_obj = template_key[good_matches[i].queryIdx]
        obj.append(temp_obj.pt)
        temp_scene_pts = scene_key[good_matches[i].trainIdx]
        scene_pts.append(temp_scene_pts.pt)

    H = cv2.findHomography(np.array(obj), np.array(scene_pts), cv2.RANSAC)
    return H, template_key, template_descript, scene_key, scene_descript, good_matches




def run_tti():
    count = 0
    template = cv2.imread("template.jpg")
    template_key, template_descript = surf.detectAndCompute(template, None)
    z = []

    for j in range(len(files) - m):
        file = files[j]
        frame = cv2.imread(os.path.join(PATH, file))
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # center = [gray.shape[0] / 2, gray.shape[1] / 2]
        # gray = gray[0:240, 250:400]

        count += 1

        H, template_key, template_descript, scene_key, scene_descript, good_matches = my_Homography2(template, frame, template_key, template_descript)
        scene = frame
        img_matches = np.empty(
            (max(template.shape[0], scene.shape[0]), template.shape[1] + scene.shape[1], 3),
            dtype=np.uint8)
        cv2.drawMatches(template, template_key, scene, scene_key, good_matches, img_matches,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        obj_corners = []
        obj_corners.append((0, 0))
        obj_corners.append((template.shape[1], 0))
        obj_corners.append((0, template.shape[0]))
        obj_corners.append((template.shape[1], template.shape[0]))

        obj_corners = np.array([obj_corners], dtype=np.float32)
        scene_corners = cv2.perspectiveTransform(obj_corners, H[0])
        scene_corners = scene_corners.astype(int)

        cv2.line(img_matches, tuple(scene_corners[0][0] + (template.shape[1], 0)), tuple(scene_corners[0][1] + (template.shape[1], 0)), (255, 255, 0), 4)
        cv2.line(img_matches, tuple(scene_corners[0][2] + (template.shape[1], 0)), tuple(scene_corners[0][3] + (template.shape[1], 0)), (255, 255, 0), 4)
        cv2.line(img_matches, tuple(scene_corners[0][0] + (template.shape[1], 0)), tuple(scene_corners[0][2] + (template.shape[1], 0)), (255, 255, 0), 4)
        cv2.line(img_matches, tuple(scene_corners[0][1] + (template.shape[1], 0)), tuple(scene_corners[0][3] + (template.shape[1], 0)), (255, 255, 0), 4)
        cv2.imshow("Matches", img_matches)
        # cv2.imwrite("video/frame-" + str(count).zfill(4) + ".jpg", img_matches)
        cv2.waitKey(1)
        object_width = np.mean([np.abs(scene_corners[0][0] - scene_corners[0][2]), np.abs(scene_corners[0][1] - scene_corners[0][3])])

        z.append(fx * can_width / object_width)

    return z


my_z = run_tti()
frame_num = np.arange(1, 18)
plt.scatter(frame_num, my_z)
plt.show()

