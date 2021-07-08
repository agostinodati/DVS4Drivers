import cv2
import os
from dv import AedatFile
import numpy as np
import math
import mediapipe as mp
from landmark_indexes import all_landmarks


def view_aedat_videoframes(file):
    '''

    :param file:
    :return:
    '''
    old_ts = 0
    with AedatFile(file) as f:
        # loop through the "frames" stream
        i = 0
        for frame in f['frames']:
            cv2.imshow('Video frames', frame.image)
            cv2.waitKey(1)
            print(frame.timestamp-old_ts)
            old_ts = frame.timestamp
            i += 1
        print(i)


def find_landmarks(video_frame, event_frame, blur=False, inverse_order=False):
    """
        This function finds face's landmarks of the i-frame.
    """

    if blur:
        video_frame = cv2.GaussianBlur(video_frame, (3, 3), 0)
    if blur and event_frame is not None:
        event_frame = cv2.GaussianBlur(event_frame, (3, 3), 0)

    mp_face_mesh = mp.solutions.face_mesh
    height, width = video_frame.shape[:2]

    if inverse_order:
        first = event_frame
        second = video_frame
    else:
        first = video_frame
        second = event_frame

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.001,
            min_tracking_confidence=0.01) as face_mesh:

        # the BGR image to RGB.
        first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        first.flags.writeable = False
        results = face_mesh.process(first)

        # is video work when inverse_order is True
        is_video = False
        if not results.multi_face_landmarks:
            second = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
            second.flags.writeable = False
            results = face_mesh.process(second)
            is_video = True
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = np.empty((len(all_landmarks), 2), np.float32)
                i = 0
                for index in all_landmarks:
                    features[i, 0] = face_landmarks.landmark[index].x * width
                    features[i, 1] = face_landmarks.landmark[index].y * height
                    i += 1
            return features, is_video
        else:
            return None, is_video


def find_landmarks_only_video(video_frame, blur=False):
    """
        This function finds face's landmarks of the i-frame.
    """

    if blur:
        video_frame = cv2.GaussianBlur(video_frame, (3, 3), 0)

    mp_face_mesh = mp.solutions.face_mesh
    height, width = video_frame.shape[:2]

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.001,
            min_tracking_confidence=0.01) as face_mesh:

        # the BGR image to RGB.
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        video_frame.flags.writeable = False
        results = face_mesh.process(video_frame)

        # is video work when inverse_order is True
        is_video = True
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = np.empty((len(all_landmarks), 2), np.float32)
                i = 0
                for index in all_landmarks:
                    features[i, 0] = face_landmarks.landmark[index].x * width
                    features[i, 1] = face_landmarks.landmark[index].y * height
                    i += 1
            return features, is_video
        else:
            return None, is_video


def optical_flow(old_event_frame, new_event_frame, landmarks, winSize=57):
    mask = np.zeros_like(old_event_frame)
    lk_params = dict(winSize=(winSize, winSize),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_event_frame, new_event_frame, landmarks, None, **lk_params)
    return p1


def draw_landmarks_optical_flow(old_landmarks, new_landmarks,video_frame, landmarks_true):
    avg = None
    # draw the tracks
    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(video_frame)
    error_sum = 0
    if landmarks_true is not None:
        for i, (new, old, true) in enumerate(zip(new_landmarks, old_landmarks, landmarks_true)):
            a, b = new.ravel()
            c, d = old.ravel()
            e, f = true.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)
            frame = cv2.circle(video_frame, (int(a), int(b)), 2, (255, 255, 255), -1)
            frame = cv2.circle(frame, (int(e), int(f)), 2, (0, 0, 255), -1)
            error_sum += math.sqrt(math.pow((a-e), 2) + math.pow((b - f), 2))
        avg = error_sum / len(landmarks_true)
        write_error_img(avg, frame)
    else:
        for i, (new, old) in enumerate(zip(new_landmarks, old_landmarks)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)
            frame = cv2.circle(video_frame, (int(a), int(b)), 2, (255, 255, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Optical flow', img)
    return avg


def write_error_img(error, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 240)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 2

    cv2.putText(img, 'Average error: {0:.2f}'.format(error),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)