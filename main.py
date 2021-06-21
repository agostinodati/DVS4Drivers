from dv import NetworkEventInput
from dv import AedatFile
import cv2
import numpy as np
import mediapipe as mp
import os.path
import utility

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

zero_landmark = rightEyeUpper1 + rightEyeLower1 + leftEyeUpper1 + leftEyeLower1

all_landmark = lipsUpperInner + lipsLowerInner + zero_landmark


def main1():
    with AedatFile("D:/Utorrent/dvSave-2021_05_28_18_48_58.aedat4") as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size
        f['events'].n
        '''
        # loop through the "events" stream
        for e in f['events']:
            print(e.timestamp)
        
        # loop through the "events" stream as numpy packets
        for e in f['events'].numpy():
            print(e.shape)'''

        # loop through the "frames" stream
        '''i = 0
        for frame in f['frames']:
            cv2.imshow('out', frame.image)
            cv2.waitKey(1)
            i += 1
        print(i)'''

        events = np.hstack([packet for packet in f['events'].numpy()])

        normalize = True  # For normalization relative to timestamps
        k = 0  # Event counter
        s = 1  # Frame counter
        time = 1000  # for 100 fps -> 1000 us
        ts = events[0]['timestamp']
        event_frame = np.zeros((height, width, 3), np.uint8)
        # event_frame = np.zeros((height, width, 1), np.uint8)
        # event_frame[:,:,0] = 127
        while k != len(events):

            # 1 millisecond skip for each frame (100 fps video)
            # All events in this time window are combined into one frame
            while k != len(events) and events[k]['timestamp'] < ts + s * time:
                e = events[k]
                k += 1
                if normalize:
                    norm_factor = (ts + s * time - e['timestamp']) / time
                else:
                    norm_factor = 1

                if e['polarity'] == 1:
                    event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
                    # event_frame[e['y'], e['x']] = int(127 * norm_factor) + 127
                else:
                    event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
                    # event_frame[e['y'], e['x']] = 127-int(127 * norm_factor)

            s += 1
            cv2.imshow('out3', event_frame)
            # Frame reset
            event_frame = np.zeros((height, width, 3), np.uint8)
            # event_frame[:, :, :] = 0
            cv2.waitKey(1)

        print(k)
        print(s)


def main2():
    with AedatFile("D:/Download/dvSave-2021_04_23_13_45_03.aedat4") as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size

        normalize = False  # For normalization relative to timestamps
        start = 0
        k = 0  # Event counter
        s = 1  # Frame counter
        time = 16000  # for 100 fps -> 1000 us
        event_frame = np.zeros((height, width, 3), np.uint8)
        video_frame = f['frames'].__next__()
        annotated_image = find_landmarks_frame(video_frame)
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
                    event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
                    # event_frame[e['y'], e['x']] = int(127 * norm_factor) + 127
                else:
                    event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
                    # event_frame[e['y'], e['x']] = 127-int(127 * norm_factor)
                k += 1

                # 1 millisecond skip for each frame (100 fps video)
                # All events in this time window are combined into one frame
                if e['timestamp'] > ts + s * time:
                    cv2.imshow('Events', event_frame)
                    cv2.imshow('Video', video_frame.image)
                    cv2.imshow('Facemesh', annotated_image)
                    if video_frame.timestamp < ts + s * time:
                        video_frame = f['frames'].__next__()
                        annotated_image = find_landmarks_frame(video_frame)
                    s += 1

                    # Frame reset
                    event_frame = np.zeros((height, width, 3), np.uint8)
                    cv2.waitKey(1)

        print(k)
        print(s)


def main3():
    # D:/Download/dvSave-2021_04_23_13_45_03.aedat4
    print("Welcome!")
    filepath = input('*.Aedat4 file path: ')
    while not os.path.isfile(filepath):
        print('File does not exists, try again.')
        filepath = input('*.Aedat4 file path: ')

    with AedatFile(filepath) as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size

        find_landmarks_frames(f['frames'])

        normalize = False  # For normalization relative to timestamps
        k = 0  # Event counter
        s = 1  # Frame counter
        time = 1000  # for 100 fps -> 1000 us
        event_frame = np.zeros((height, width, 3), np.uint8)
        for packet in f['events'].numpy():
            for e in packet:

                if k == 0:
                    ts = e['timestamp']

                if normalize:
                    norm_factor = (ts + s * time - e['timestamp']) / time
                else:
                    norm_factor = 1

                if e['polarity'] == 1:
                    event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
                    # event_frame[e['y'], e['x']] = int(127 * norm_factor) + 127
                else:
                    event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
                    # event_frame[e['y'], e['x']] = 127-int(127 * norm_factor)
                k += 1

                # 1 millisecond skip for each frame (100 fps video)
                # All events in this time window are combined into one frame
                if e['timestamp'] > ts + s * time:
                    s += 1
                    cv2.imshow('out3', event_frame)
                    # Frame reset
                    event_frame = np.zeros((height, width, 3), np.uint8)
                    cv2.waitKey(1)

        print(k)
        print(s)


def find_landmarks_frame(frame):
    """
        This function finds face's landmarks of the i-frame.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    height, width = frame.size

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        image = frame.image

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                '''mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)'''
                # rightEyeUpper0: [246, 161, 160, 159, 158, 157, 173],
                for index in all_landmark:
                    image2 = cv2.circle(image, (
                    int(face_landmarks.landmark[index].x * height), int(face_landmarks.landmark[index].y * width)),
                                        radius=0, color=(0, 0, 255), thickness=-1)
        return image2


def find_landmarks_frames(frames):
    """
        Not used.
        This function finds face's landmarks for a list of frames.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        for frame in frames:
            image = frame.image

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            cv2.imshow('MediaPipe FaceMesh', image)
            cv2.waitKey(1)
            if cv2.waitKey(5) & 0xFF == 27:
                break


if __name__ == '__main__':
    main2()
