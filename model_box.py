import cv2 as cv
import numpy as np


def find_model_box(frame, number_point, sift, bf, des_model, min_matches):
    X, Y, W, H = None, None, None, None

    # Detect edges in image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to gray
    gray = cv.GaussianBlur(gray, (5, 5), 0)  # Blur the image
    edges = cv.Canny(gray, 100, 200)  # Detect edges

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    match_find = 0
    find = False

    for contour in contours:
        # Shape check
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.01 * perimeter, True)

        if len(approx) == number_point:  # If we have 4 points
            x, y, w, h = cv.boundingRect(approx)  # Get model in the frame
            mini_model = cv.cvtColor(frame[y:y + h, x:x + w], cv.COLOR_BGR2GRAY)

            # find and draw the keypoints of the frame
            kp_frame, des_frame = sift.detectAndCompute(mini_model, None)

            if des_frame is not None:
                # match frame descriptors with model descriptors
                matches = bf.match(np.uint8(des_model), np.uint8(des_frame))
                # sort them in the order of their distance
                # the lower the distance, the better the match
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > match_find and len(matches) >= min_matches:
                    match_find = len(matches)
                    X, Y, W, H = x, y, w, h
                    find = True

    if not find:
        X, Y, W, H = None, None, None, None

    return X, Y, W, H