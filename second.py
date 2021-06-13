from __future__ import print_function
import argparse
import datetime
import imutils
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
                help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
                help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.05,
                help="image pyramid scale")
ap.add_argument("-m", "--mean-shift", type=int, default=-1,
                help="whether or not mean shift grouping should be used")
args = vars(ap.parse_args())

winStride = eval(args["win_stride"])
padding = eval(args["padding"])
meanShift = True if args["mean_shift"] > 0 else False

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(400, image.shape[1]))
#
# start = datetime.datetime.now()
# (rects, weights) = hog.detectMultiScale(image, winStride=winStride, hitThreshold=0.2,
#                                         padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
# print("[INFO] detection took: {}s".format(
#     (datetime.datetime.now() - start).total_seconds()))
#
# for (x, y, w, h) in rects:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imshow("Detections", image)
# cv2.waitKey(0)


orig = image.copy()
# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), scale=1.05, hitThreshold=0.2)
# draw the original bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)
# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
# show some information on the number of bounding boxes

print("[INFO] {}: {} original boxes, {} after suppression".format(
    'filename', len(rects), len(pick)))
# show the output images
cv2.imshow("Before NMS", orig)
cv2.imshow("After NMS", image)
cv2.waitKey(0)
