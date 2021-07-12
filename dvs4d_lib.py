import cv2
from dv import AedatFile
import numpy as np
import math
import mediapipe as mp
from landmark_indexes import all_landmarks, left_eye, right_eye, mouth
import math


def view_aedat_videoframes(file):
    '''
    Show the video frames of the Aedat file.
    :param file: Path of the aedat file
    '''
    old_ts = 0
    with AedatFile(file) as f:
        # loop through the "frames" stream
        i = 0
        for frame in f['frames']:
            cv2.imshow('Video frames', frame.image)
            cv2.waitKey(1)
            print(frame.timestamp - old_ts)
            old_ts = frame.timestamp
            i += 1
        print(i)


def find_landmarks(video_frame, event_frame, blur=True, inverse_order=False):
    '''
    Calculate the landmarks of the face.
    First try on the event frame (if inverse_order is True).
    :param video_frame: Video frame
    :param event_frame: Event frame
    :param blur: If True, blur the event frame for a better interpretation
    :param inverse_order: If True, make the first try on the event frame
    :return: Landmarks and a flag (is_video)
    '''

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
    '''
    Calculate the landmarks of the face from the video frame.
    :param video_frame: Video frame
    :param blur: If True, blur the event frame for a better interpretation
    :return: Landmarks and a flag (is_video)
    '''

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
    '''
    Calculate the optical flow to make an estimate of the movement of the landmarks
    :param old_event_frame: Previous event frame
    :param new_event_frame: Next event frame
    :param landmarks: Landmarks of the face calculated using facemesh
    :param winSize: Size of the window used by OpenCv
    :return: Estimated position of landmarks
    '''
    mask = np.zeros_like(old_event_frame)
    lk_params = dict(winSize=(winSize, winSize),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_event_frame, new_event_frame, landmarks, None, **lk_params)
    return p1


def optical_flow_farneback(old_event_frame, new_event_frame, landmarks, winSize=21):
    '''
    Calculate the optical flow to make an estimate of the movement of the landmarks
    :param old_event_frame: Previous event frame
    :param new_event_frame: Next event frame
    :param landmarks: Landmarks of the face calculated using facemesh
    :param winSize: Size of the window used by OpenCv
    :return: Estimated position of landmarks
    '''
    flow = cv2.calcOpticalFlowFarneback(old_event_frame, new_event_frame, None, 0.5, 3, winSize, 3, 7, 1.5, 0)
    p1 = landmarks.copy()
    for i in range(landmarks.shape[0]):
        p1[i][0] += flow[round(landmarks[i][1]), round(landmarks[i][0])][0]
        p1[i][1] += flow[round(landmarks[i][1]), round(landmarks[i][0])][1]
    return p1


def draw_landmarks_optical_flow(old_landmarks, new_landmarks, video_frame, landmarks_true):
    '''
    Draw the estimated landmarks of the optical flow and calculate the error of the estimation.
    :param old_landmarks: Landmarks of the previous frame
    :param new_landmarks: Landmarks of the next frame
    :param video_frame: Video frame
    :param landmarks_true: Landmarks of the next frame calculated on the video frame.
    :return: Average error (euclidean distance between estimated landmarks and true landmarks calculate using facemesh on video frame)
    '''
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
            error_sum += math.sqrt(math.pow((a - e), 2) + math.pow((b - f), 2))  # Euclidean distance
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
    '''
    Write the error on the image.
    :param error: Float
    :param img: Image
    '''
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


def face_roi(landmarks, frame1, frame2, offset=30):
    '''
    Exatract the ROI of the face using the landmarks, calculated using facemesh.
    :param landmarks: Facemesh's landmarks
    :param frame1: Previous image
    :param frame2: Next image
    :param offset: ROI's offeset
    :return: ROIs of Previous and Next images
    '''
    height, width = frame1.shape[:2]
    if landmarks is not None:
        minx = width
        miny = height
        maxy = 0
        maxx = 0
        for landmark in landmarks:
            x = int(landmark[0])
            y = int(landmark[1])
            if x < minx:
                minx = x
            if y < miny:
                miny = y
            if x > maxx:
                maxx = x
            if y > maxy:
                maxy = y

        if minx - offset > 0:
            minx -= offset
        else:
            minx = 0

        if miny - offset > 0:
            miny -= offset
        else:
            miny = 0

        if maxx + offset < width:
            maxx += offset
        else:
            maxx = width

        if maxy + offset < height:
            maxy += height
        else:
            maxy = height

        w = maxx - minx
        h = maxy - miny
        if w > 0 and h > 0:
            black_frame1 = np.zeros((height, width, 1), np.uint8)
            black_frame2 = black_frame1.copy()
            black_frame1[miny:maxy, minx:maxx] = frame1[miny:maxy, minx:maxx]
            black_frame2[miny:maxy, minx:maxx] = frame1[miny:maxy, minx:maxx]
            return black_frame1, black_frame2
    return frame1, frame2


def naive_event_drawer(normalize, event, frame, dt=1, endTs=0):
    '''
    Draw the events in a simple way.
    :param normalize: A flag used for the normalization
    :param event: Current event
    :param frame: Frame where the event will be draw
    :param dt: Temporal window
    :param endTs:
    :return: Frame with the event drawn
    '''
    if normalize:
        norm_factor = (endTs - event[0]) / dt
    else:
        norm_factor = 1

    if event[3] == 1:
        # event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
        frame[event[2], event[1]] = int(127 * norm_factor) + 127
    else:
        # event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
        frame[event[2], event[1]] = 127 - int(127 * norm_factor)
    return frame


def accumulator(event, frame, increment=30):
    '''
    Draw the accumulator of the events. It generates a sophisticate representation of the events
    :param event: Current event
    :param frame: Frame where the event will be draw
    :param increment: Increment of intensity
    :return: The accumulator frame
    '''
    if event[3] == 1:
        if frame[event[2], event[1]] < 255 - increment:
            frame[event[2], event[1]] += increment
        else:
            frame[event[2], event[1]] = 255
    else:
        if frame[event[2], event[1]] > 0 + increment:
            frame[event[2], event[1]] -= increment
        else:
            frame[event[2], event[1]] = 0
    return frame


def extract_eye_mouth_rois(frame):
    '''
    Extract the eyes and mouth rois of the face in the image passed.
    :param frame: Image
    :return: Left eye roi, Right eye roi, Mouth roi
    '''
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    height, width = frame.shape[:2]

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as face_mesh:

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_roi = extract_roi_coord(width, height, face_landmarks.landmark, left_eye)
                right_eye_roi = extract_roi_coord(width, height, face_landmarks.landmark, right_eye)
                mouth_roi = extract_roi_coord(width, height, face_landmarks.landmark, mouth)
            return left_eye_roi, right_eye_roi, mouth_roi
        else:
            return None, None, None


def extract_roi_coord(width, height, landmarks, indexes):
    '''
    Exatract the coordinates of the roi.
    :param width: Width of the image
    :param height: Height of the image
    :param landmarks: Facemesh's landmarks
    :param indexes: Index of landmarks (of the facemesh's mask) for that specific roi.
    :return: Coordinates of the roi.
    '''
    minx = width
    miny = height
    maxy = 0
    maxx = 0
    for index in indexes:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
    w = maxx - minx
    h = maxy - miny
    return minx, maxx, miny, maxy


def detect_blink(eye_event_frame, image, threshold_low, threshold_up):
    '''
    Detect the blinking (up and down) of the eye using the variation of the events.
    :param eye_event_frame: Event's image of the eye
    :param image: Complete image
    :param threshold_low: Threshold for blink up
    :param threshold_up: Threshold for blink down
    '''
    height, width = eye_event_frame.shape[:2]
    h1 = int(height/2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 240)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 2
    upper_roi = eye_event_frame[0:h1, 0:width]
    lower_roi = eye_event_frame[h1:height, 0:width]
    avg_upper_roi = np.mean(upper_roi)
    avg_lower_roi = np.mean(lower_roi)

    ratio = avg_upper_roi/avg_lower_roi

    low_eye = cv2.resize(lower_roi, (width * 5, h1 * 5))
    up_eye = cv2.resize(upper_roi, (width * 5, h1 * 5))
    print('Average lower values: ' + str(avg_lower_roi))
    print('Average upper values: ' + str(avg_upper_roi))
    print('Ratio: ' + str(ratio))
    cv2.imshow('Lower eye', low_eye)
    cv2.imshow('Upper eye', up_eye)
    if ratio >= threshold_up:
        cv2.putText(image, 'Blink down',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    elif ratio <= threshold_low:
        cv2.putText(image, 'Blink up',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)


def detect_mouth_opening(mouth_roi, image, treshold=0.4):
    '''
    Detect the opening of the mouth using the variation of the roi.
    :param mouth_roi: Roi of the mouth
    :param image: Image
    :param treshold: Treshold
    '''
    height, width = mouth_roi.shape[:2]
    ratio = height/width
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 240)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 2
    if ratio > treshold:
        cv2.putText(image, 'Mouth Open',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

