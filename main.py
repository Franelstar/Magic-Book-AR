import argparse
import cv2 as cv
from model_box import find_model_box
import numpy as np
import os
from detection import detect
from tensorflow.keras.applications.inception_v3 import InceptionV3
from objects import list_objects
from projection import *

# matrix of camera parameters (made up but works quite well for me)
camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
QUADRILATERAL_POINTS = 4
MIN_MATCHES = 4
FLANN_INDEX_KDTREE = 1

model_inception_v3 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                                 pooling=None, classes=1000)

ignore = 0
X, Y, W, H = None, None, None, None
homography = None
objet, proba = ('', 0.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--camera", required=False, help="Camera id", default=0, type=int)
    parser.add_argument("-r", "--rectangle", required=False, help="View rectangle", default=False, type=bool)
    parser.add_argument("-m", "--matches", required=False, help="View matches", default=False, type=bool)

    args = vars(parser.parse_args())
    camera_id = args['camera']
    view_rectangle = args['rectangle']
    view_matches = args['matches']

    sift = cv.SIFT_create()

    # create BFMatcher object based on hamming distance
    bf = cv.BFMatcher()

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv.imread(os.path.join(dir_name, 'reference/model.jpg'), 0)

    # Compute model keypoints and its descriptors
    kp_model, des_model = sift.detectAndCompute(model, None)

    video_capture = cv.VideoCapture(camera_id)

    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # fourcc2 = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('simple.avi', fourcc, 20.0, (640, 480))
    # out2 = cv.VideoWriter('output.avi', fourcc2, 20.0, (900, 480))

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if ret and frame is not None:
            # Get model box in frame
            X, Y, W, H = find_model_box(frame, QUADRILATERAL_POINTS, sift, bf, des_model, MIN_MATCHES)

            if ignore % 25 == 0:  # Any second
                objet, proba = detect(frame, model_inception_v3)  # Get object and probability of the model prediction
                ignore = 0
            ignore += 1

            if proba > 0.1 and objet in list(list_objects.keys()):
                # Load 3D model from OBJ file
                obj = list_objects[objet][0]

                # Display name's object
                cv.putText(frame, list_objects[objet][5], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 10, 5), 2)

                if X is not None:
                    # find and draw the keypoints of the frame
                    kp_frame, des_frame = sift.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)

                    # Draw Keys points
                    # frame = cv.drawKeypoints(frame, kp_frame, 0, color=(0, 255, 0), flags=0)

                    if des_frame is not None:

                        # match frame descriptors with model descriptors
                        matches = flann.knnMatch(np.float32(des_model), np.float32(des_frame), k=2)

                        # Vérification de la robustesse des différents keypoints
                        good = []
                        for m, n in matches:
                            try:
                                if m.distance < 0.7 * n.distance:
                                    point_frame = kp_frame[m.trainIdx].pt
                                    point_model = kp_model[m.trainIdx].pt
                                    # Si le point est dans le cadre ...
                                    if X <= point_frame[0] <= X + W and Y <= point_frame[1] <= Y + H:
                                        good.append(m)
                            except:
                                pass

                        if len(good) >= MIN_MATCHES:
                            src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                            # compute Homography
                            homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                            if view_rectangle:
                                # Draw a rectangle that marks the found model in the frame
                                h, w = model.shape
                                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                                # project corners into frame
                                dst = cv.perspectiveTransform(pts, homography)
                                # connect them with lines
                                frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

                            # if a valid homography matrix was found render cube on model plane
                            if homography is not None:
                                try:
                                    # obtain 3D projection matrix from homography matrix and camera parameters
                                    projection = projection_matrix(camera_parameters, homography)

                                    # Rotation
                                    r = np.radians(list_objects[objet][1])
                                    rot = rotation_matrix(r)
                                    projection = np.dot(projection, rot)

                                    # project cube or model
                                    frame = render(np.uint8(frame), obj, projection, model, list_objects[objet][3],
                                                   list_objects[objet][2], list_objects[objet][4])
                                except:
                                    pass

                                # draw first 10 matches.
                                if view_matches:
                                    matchesMask = mask.ravel().tolist()
                                    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                                       singlePointColor=None,
                                                       matchesMask=matchesMask,  # draw only inliers
                                                       flags=2)
                                    frame = cv.drawMatches(np.uint8(model), kp_model, np.uint8(frame),
                                                           kp_frame, good, None, **draw_params)
            # Write video
            # out.write(frame2)
            # out2.write(frame)

            # display frame
            cv.imshow('frame', frame)

            # Pour quitter, on tappe sur la touche q
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            if cv.waitKey(1) & 0xFF == ord('s'):
                cv.imwrite('r4.png', frame)
        else:
            print('Impossible d\'acceder à la camera')
            break

    # Release handle to the webcam
    video_capture.release()
    cv.destroyAllWindows()
    # out.release()
    # out2.release()
