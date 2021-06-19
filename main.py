from dv import NetworkEventInput
from dv import AedatFile
import cv2
import numpy as np
import mediapipe as mp
import os.path


def main1():
    with AedatFile("D:/Utorrent/dvSave-2021_05_28_18_48_58.aedat4") as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size

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
    with AedatFile("D:/Utorrent/dvSave-2021_05_28_18_48_58.aedat4") as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size

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

        # find_landmarks_static(f['frames'])

        path_video = frame2avi(f['frames'])

        find_landmarks_frames(path_video)

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


def check_dir(path):
    if not os.path.exists(path):
        print('Directory ' + '"if os.path.exists(path):"' + ' does not exists. /nCreating the dir...')
        os.makedirs(path)
    if not os.path.isdir(path):
        print('The path indicated is not a directory.')
        return False
    return True


def find_landmarks_static(frames):
    '''
        :param frames: List of Frames of the Aedat4 file.

        This function take the first frame of the aedat4 file and save it as image. Then, find the face landmarks.
    '''
    print('You are in the Static mode.')

    path = input('Path where to save the img of the video: ')
    while not check_dir(path):
        path = input('Path where to save the img of the video: ')

    path_results = input('Path where to save the img of the results: ')
    while not check_dir(path):
        path_results = input('Path where to save the img of the results: ')

    save_path = path + '/frame.jpg'

    for frame in frames:
        cv2.imwrite(save_path, frame.image)
        break

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    image_files = [save_path]

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(image_files):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print(results.multi_face_landmarks)
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
            cv2.imwrite(path_results + '/annotated_image' + str(idx) + '.png', annotated_image)
            cv2.imshow('out', annotated_image)
            cv2.waitKey(1)


def frame2avi(frames):
    '''
        :param frames: List of Frames of the Aedat4 file.
        :return: Path of the created video.

        This function create a video (*.avi) from the Frames of the Aedat4 file.
    '''
    # D:/openCV
    path = input('Path where to save the *.avi: ')
    while not check_dir(path):
        path = input('Path where to save the *.avi: ')
    path_frames = path + '/frames'
    name = '/videoTest.avi'
    path_video = path + name

    if os.path.isfile(path_video):
        print('The file' + '"' + path_video + '"' + ' already exists.')
        return path_video

    codec = 0
    fps = 25
    height, width = frames.size
    size = (width, height)
    out = cv2.VideoWriter(path_video, codec, fps, size)
    i = 0
    print('Creating the *.avi...')
    for frame in frames:
        cv2.imwrite(path_frames + '/frame_' + str(i) + '.jpg', frame.image)
        img = cv2.imread(path_frames + '/frame_' + str(i) + '.jpg')
        out.write(img)
        i = i + 1
        print('...')
    cv2.destroyAllWindows()
    out.release()
    print('*.avi created!')
    return path_video


def find_landmarks_frames(path_video):
    '''
        :param path_video: Path of the *.avi file.

        This function finds face's landmarks.
    '''
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(path_video)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                #continue
                break

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
    cap.release()


if __name__ == '__main__':
    main3()
