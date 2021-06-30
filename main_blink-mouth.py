from dv import NetworkEventInput
from dv import NetworkFrameInput
from dv import AedatFile
import cv2
import numpy as np
import mediapipe as mp
import os.path
import utility

silhouette = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
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

all_landmarks = left_eye + right_eye + mouth+ silhouette

amal1 = "C:/Users/User/Downloads/dvSave-2021_04_23_13_45_03.aedat4"
amal2 = "D:/Utorrent/dvSave-2021_05_28_18_48_58.aedat4"
ago1 = "D:/Download/mancini.aedat4"
ago2 = "D:/Download/mancini_notte.aedat4"


def main():
    with AedatFile(ago1) as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size

        normalize = False  # For normalization relative to timestamps
        start = 0
        k = 0  # Event counter
        s = 1  # Frame counter
        time = 8000  # for 100 fps -> 10000 us
        event_frame = np.zeros((height, width, 1), np.uint8)
        event_frame[:, :, 0] = 127
        old_event_frame = event_frame
        video_frame = f['frames'].__next__()
        annotated_image = find_landmarks_frame(video_frame.image, video_frame.image)
        for packet in f['events'].numpy():
            for e in packet:

                '''if k < start: # Per iniziare da un frame diverso da quello iniziale del video
                    k += 1
                    if video_frame['timestamp'] < e['timestamp']:
                        video_frame = f['frames'].__next__()
                    continue'''

                if k == start:
                    ts = e['timestamp']

                if normalize:
                    norm_factor = (ts + s * time - e['timestamp']) / time
                else:
                    norm_factor = 1

                if e['polarity'] == 1:
                    # event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
                    event_frame[e['y'], e['x']] = int(127 * norm_factor) + 127
                else:
                    # event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
                    event_frame[e['y'], e['x']] = 127-int(127 * norm_factor)
                k += 1

                # 1 millisecond skip for each frame (100 fps video)
                # All events in this time window are combined into one frame
                if e['timestamp'] > ts + s * time:
                    cv2.imshow('Events', event_frame)
                    cv2.imshow('Video', video_frame.image)
                    cv2.imshow('Facemesh', annotated_image)

                    while video_frame.timestamp < ts + s * time:
                        annotated_image = find_landmarks_frame(event_frame, video_frame.image)
                        video_frame = f['frames'].__next__()
                    s += 1

                    # Frame reset
                    old_event_frame = event_frame
                    event_frame = np.zeros((height, width, 1), np.uint8)
                    event_frame[:, :, 0] = 127
                    cv2.waitKey(1)

        print(k)
        print(s)


def main2():
    with NetworkEventInput(address='127.0.0.1', port=9999) as ev, \
            NetworkFrameInput(address='127.0.0.1', port=8888) as f:

        height = 260
        width = 346

        normalize = True  # For normalization relative to timestamps

        start = 0
        k = 0  # Event counter
        s = 1  # Frame counter
        time = 8000  # for 100 fps -> 10000 us
        event_frame = np.zeros((height, width, 1), np.uint8)
        event_frame[:, :, 0] = 0
        old_event_frame = event_frame
        video_frame = f.__next__()
        print(video_frame.timestamp)

        annotated_image = find_landmarks_frame(video_frame.image, video_frame.image)
        for e in ev:

            '''if k < start: # Per iniziare da un frame diverso da quello iniziale del video
                k += 1
                if video_frame['timestamp'] < e['timestamp']:
                    video_frame = f['frames'].__next__()
                continue'''

            if k == start:
                ts = e.timestamp

            if normalize:
                norm_factor = (ts + s * time - e['timestamp']) / time
            else:
                norm_factor = 1

            if e.polarity == 1:
                # event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
                event_frame[e.y, e.x] = int(127 * norm_factor) + 127
            else:
                # event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
                event_frame[e.y, e.x] = 127 - int(127 * norm_factor)
            k += 1


            # 1 millisecond skip for each frame (100 fps video)
            # All events in this time window are combined into one frame
            if e.timestamp > ts + s * time:
                cv2.imshow('Events', event_frame)
                cv2.imshow('Video', video_frame.image)
                cv2.imshow('Facemesh', annotated_image)

                while video_frame.timestamp < ts + s * time:
                    annotated_image = find_landmarks_frame(event_frame, video_frame.image)
                    video_frame = f.__next__()
                s += 1

                # Frame reset
                old_event_frame = event_frame
                event_frame[:, :, 0] = 127
                cv2.waitKey(1)

        print(k)
        print(s)


def find_landmarks_frame(image, video_frame):
    """
        This function finds face's landmarks of the i-frame.
    """
    image_blurred = cv2.GaussianBlur(image, (9, 9), 0)
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    height, width = image.shape[:2]

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_blurred.flags.writeable = False
        results = face_mesh.process(image_blurred)

        # Draw the face mesh annotations on the image.
        image_blurred.flags.writeable = True
        image2 = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2BGR)
        if not results.multi_face_landmarks:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_frame.flags.writeable = False
            image2 = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
            results = face_mesh.process(video_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                '''mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)'''
                image2 = utility.draw_landmarks(width, height, image2, face_landmarks.landmark, left_eye, image, 'Left Eye')
                image2 = utility.draw_landmarks(width, height, image2, face_landmarks.landmark, right_eye, image, 'Right Eye')
                image2 = utility.draw_landmarks(width, height, image2, face_landmarks.landmark, mouth, image, 'Mouth')
                image2 = utility.draw_landmarks(width, height, image2, face_landmarks.landmark, silhouette, image, 'Silhouette')
        return image2


if __name__ == '__main__':
    main()
