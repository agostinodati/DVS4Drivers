from dv import NetworkEventInput
from dv import NetworkFrameInput
from dv import AedatFile
import cv2
import numpy as np
import mediapipe as mp
import os.path
import utility
import random

silhouette = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
rightEyeUpper2 = [113, 225, 224, 223, 222, 221, 189]
rightEyeLower2 = [226, 31, 228, 229, 230, 231, 232, 233, 244]
rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]

leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
leftEyeUpper2 = [342, 445, 444, 443, 442, 441, 413]
leftEyeLower2 = [446, 261, 448, 449, 450, 451, 452, 453, 464]
leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]

eye_landmark = rightEyeUpper1 + rightEyeLower1 + leftEyeUpper1 + leftEyeLower1

left_eye = leftEyeUpper1 + leftEyeLower1
right_eye = rightEyeUpper1 + rightEyeLower1
mouth = lipsUpperInner + lipsLowerInner

all_landmarks = left_eye + right_eye + mouth + silhouette

amal1 = "C:/Users/User/Downloads/dvSave-2021_04_23_13_45_03.aedat4"
amal2 = "D:/Utorrent/dvSave-2021_05_28_18_48_58.aedat4"
ago1 = "D:/Download/mancini.aedat4"
ago2 = "D:/Download/mancini_notte.aedat4"


def main_optical_flow():
    with AedatFile(amal1) as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size

        normalize = True  # For normalization relative to timestamps
        start = 0
        k = 0  # Event counter
        s = 1  # Frame counter
        dt = 10000  # for 100 fps -> 10000 us
        video_dt = 39980
        delay_old_frame = 20000
        advance_new_frame = 20000

        new_event_frame = np.zeros((height, width, 1), np.uint8)
        new_event_frame[:, :, 0] = 127
        old_event_frame = new_event_frame.copy()
        video_frame = f['frames'].__next__()
        annotated_image = new_event_frame
        old_landmarks = calc_landmarks(video_frame.image)
        ts1 = video_frame.timestamp
        previous_facemesh_fail = False
        for packet in f['events'].numpy():
            for e in packet:

                ts = e['timestamp']
                if ts1 + delay_old_frame <= ts < ts1 + delay_old_frame + dt:
                    old_event_frame = utility.accumulate(normalize, e, old_event_frame, dt, ts1 + delay_old_frame + dt)
                    k += 1

                if ts1 + video_dt - advance_new_frame - dt <= ts < ts1 + video_dt - advance_new_frame:
                    new_event_frame = utility.accumulate(normalize, e, new_event_frame, dt, ts1 + video_dt - advance_new_frame)
                    k += 1

                # 1 millisecond skip for each frame (100 fps video)
                # All events in this time window are combined into one frame
                if ts >= ts1 + video_dt:
                    while video_frame.timestamp <= ts:
                        video_frame = f['frames'].__next__()
                        ts1 = video_frame.timestamp
                        cv2.imshow('Video', video_frame.image)
                        cv2.imshow('Old Event Frame', old_event_frame)
                        cv2.imshow('New Event Frame', new_event_frame)

                        new_landmarks_true = calc_landmarks(video_frame.image)
                        if random.randint(1, 10) <= 10:
                            new_landmarks = None
                        else:
                            new_landmarks = new_landmarks_true
                        facemesh_fail = False
                        if new_landmarks is None and old_landmarks is not None:
                            facemesh_fail = True
                            if facemesh_fail and previous_facemesh_fail:
                                new_landmarks, st = utility.optical_flow(previous_stored_new_frame, new_event_frame,
                                                                         old_landmarks)
                            else:
                                new_landmarks, st = utility.optical_flow(old_event_frame, new_event_frame,
                                                                         old_landmarks)
                            utility.draw_landmarks_optical_flow(old_landmarks, new_landmarks, st, video_frame.image, new_landmarks_true)
                        old_landmarks = new_landmarks

                    s += 1
                    previous_facemesh_fail = facemesh_fail
                    previous_stored_new_frame = new_event_frame.copy()
                    # Frame reset
                    new_event_frame[:, :, 0] = 127
                    old_event_frame[:, :, 0] = 127
                    cv2.waitKey(1)

        print(k)
        print(s)


def main_optical_flow2():
    with NetworkEventInput(address='127.0.0.1', port=9999) as ev, \
            NetworkFrameInput(address='127.0.0.1', port=8888) as f:
        # list all the names of streams in the file
        # print(f.names)

        # Access dimensions of the event stream
        #height, width = f.size
        height = 260
        width = 346
        normalize = False  # For normalization relative to timestamps
        start = 0
        k = 0  # Event counter
        s = 1  # Frame counter
        dt = 8000  # for 100 fps -> 10000 us
        video_dt = 39980
        delay_old_frame = 0
        advance_new_frame = 0

        new_event_frame = np.zeros((height, width, 1), np.uint8)
        new_event_frame[:, :, 0] = 127
        old_event_frame = new_event_frame.copy()
        video_frame = f.__next__()
        annotated_image = new_event_frame
        old_landmarks = calc_landmarks(video_frame.image)
        ts1 = video_frame.timestamp
        previous_facemesh_fail = False
        for e in ev:

            ts = e.timestamp
            if ts1 + delay_old_frame <= ts < ts1 + delay_old_frame + dt:
                old_event_frame = utility.accumulate_fromNetwork(e, old_event_frame)
                k += 1

            if ts1 + video_dt - advance_new_frame - dt <= ts < ts1 + video_dt - advance_new_frame:
                new_event_frame = utility.accumulate_fromNetwork(e, new_event_frame)
                k += 1

            # 1 millisecond skip for each frame (100 fps video)
            # All events in this time window are combined into one frame
            if ts >= ts1 + video_dt:
                while video_frame.timestamp <= ts:
                    video_frame = f.__next__()
                    ts1 = video_frame.timestamp
                    cv2.imshow('Video', video_frame.image)
                    cv2.imshow('Old Event', old_event_frame)
                    cv2.imshow('New Event', new_event_frame)

                    new_landmarks_true = calc_landmarks(video_frame.image)
                    if random.randint(1, 10) <= 5:
                        new_landmarks = None
                    else:
                        new_landmarks = new_landmarks_true
                    facemesh_fail = False
                    if new_landmarks is None and old_landmarks is not None:
                        facemesh_fail = True
                        if facemesh_fail and previous_facemesh_fail:
                            new_landmarks, st = utility.optical_flow(previous_stored_new_frame, new_event_frame,
                                                                     old_landmarks)
                        else:
                            new_landmarks, st = utility.optical_flow(old_event_frame, new_event_frame,
                                                                     old_landmarks)
                        utility.draw_landmarks_optical_flow(old_landmarks, new_landmarks, st, video_frame.image, new_landmarks_true)
                    old_landmarks = new_landmarks

                s += 1
                previous_facemesh_fail = facemesh_fail
                previous_stored_new_frame = new_event_frame.copy()
                # Frame reset
                new_event_frame[:, :, 0] = 127
                old_event_frame[:, :, 0] = 127
                cv2.waitKey(1)

        print(k)
        print(s)


def find_optical_flow(old_frame, curr_frame, video_frame):
    """
        This function finds face's landmarks of the i-frame.
    """
    # old_frame = cv2.GaussianBlur(old_frame, (5, 5), 0)
    # curr_frame = cv2.GaussianBlur(curr_frame, (5, 5), 0)
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    height, width = video_frame.shape[:2]

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:

        # the BGR image to RGB.
        image_blurred = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_blurred.flags.writeable = False
        results = face_mesh.process(image_blurred)

        # Draw the face mesh annotations on the image.
        image_blurred.flags.writeable = True
        image2 = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = np.empty((len(all_landmarks), 2), np.float32)
                i = 0
                for index in all_landmarks:
                    features[i, 0] = face_landmarks.landmark[index].x * width
                    features[i, 1] = face_landmarks.landmark[index].y * height
                    i += 1
                image = utility.optical_flow(old_frame, curr_frame, features)
        return image


def calc_landmarks(video_frame):
    """
        This function finds face's landmarks of the i-frame.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    height, width = video_frame.shape[:2]

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:

        # the BGR image to RGB.
        image_blurred = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_blurred.flags.writeable = False
        results = face_mesh.process(image_blurred)

        # Draw the face mesh annotations on the image.
        image_blurred.flags.writeable = True
        image2 = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = np.empty((len(all_landmarks), 2), np.float32)
                i = 0
                for index in all_landmarks:
                    features[i, 0] = face_landmarks.landmark[index].x * width
                    features[i, 1] = face_landmarks.landmark[index].y * height
                    i += 1
            return features
        else:
            return None


if __name__ == '__main__':
    main_optical_flow()
    # utility.only_video(amal2)
