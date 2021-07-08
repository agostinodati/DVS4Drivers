import cv2
import os
from dv import AedatFile
import numpy as np
import math


def check_dir(path):
    if not os.path.exists(path):
        print('Directory ' + '"if os.path.exists(path):"' + ' does not exists. /nCreating the dir...')
        os.makedirs(path)
    if not os.path.isdir(path):
        print('The path indicated is not a directory.')
        return False
    return True


def only_video(file):
    old_ts = 0
    with AedatFile(file) as f:

        # loop through the "frames" stream
        i = 0
        for frame in f['frames']:
            cv2.imshow('out', frame.image)
            cv2.waitKey(1)
            print(frame.timestamp-old_ts)
            old_ts = frame.timestamp
            i += 1
        print(i)


def optical_flow(old_event_frame, new_event_frame, landmarks):
    mask = np.zeros_like(old_event_frame)
    lk_params = dict(winSize=(57, 57),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_event_frame, new_event_frame, landmarks, None, **lk_params)
    return p1, st
    # Now update the previous frame and previous points
    # old_gray = frame_gray.copy()
    # p0 = good_new.reshape(-1, 1, 2)


def draw_landmarks(width, height, image, landmarks, indexes, source, title):
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
        image = cv2.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=2)
    w = maxx - minx
    h = maxy - miny
    im = cv2.resize(source[miny:maxy, minx:maxx], (w * 5, h * 5))
    cv2.imshow(title, im)
    return image


def face_roi(landmarks, frame1, frame2, offset=30):
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

        if minx-offset > 0:
            minx -= offset
        else:
            minx = 0

        if miny-offset > 0:
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
            # cv2.imshow('Face detection', black_frame1)
            return black_frame1, black_frame2
    return frame1, frame2


def draw_landmarks_optical_flow(old_landmarks, new_landmarks, st, video_frame, landmarks_true):
    '''    # Select good points
    if new_landmarks is not None:
        s = np.count_nonzero(st)
        good_new = np.empty((s, 2), np.float32)
        good_old = np.empty((s, 2), np.float32)
        i = 0
        k = 0
        for n in st:
            if n == 1:
                good_new[k] = new_landmarks[i]
                good_old[k] = old_landmarks[i]
                k += 1
            i += 1
    '''
    avg = None
    # draw the tracks
    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(video_frame)
    sum = 0
    if landmarks_true is not None:
        for i, (new, old, true) in enumerate(zip(new_landmarks, old_landmarks, landmarks_true)):
            a, b = new.ravel()
            c, d = old.ravel()
            e, f = true.ravel()
            # print('{0} - {1}'.format(a-c, b-d))
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)
            frame = cv2.circle(video_frame, (int(a), int(b)), 2, (255, 255, 255), -1)
            frame = cv2.circle(frame, (int(e), int(f)), 2, (0, 0, 255), -1)
            sum += math.sqrt(math.pow((a-e), 2) + math.pow((b - f), 2))
        avg = sum / len(landmarks_true)
        write_error_img(avg, frame)
    else:
        for i, (new, old) in enumerate(zip(new_landmarks, old_landmarks)):
            a, b = new.ravel()
            c, d = old.ravel()
            # print('{0} - {1}'.format(a-c, b-d))
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


def accumulate(normalize, event, frame, dt = 1, endTs = 0):
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


def naive_event_drawer(normalize, event, frame, dt = 1, endTs = 0):
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


def accumulator(event, frame, increment = 30):
    if event[3] == 1:
        # event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
        # frame[event['y'], event['x']] = int(127 * norm_factor) + 127
        if frame[event[2], event[1]] < 255 - increment:
            frame[event[2], event[1]] += increment
    else:
        # event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
        # frame[event['y'], event['x']] = 127 - int(127 * norm_factor)
        if frame[event[2], event[1]] > 0 + increment:
            frame[event[2], event[1]] -= increment

    return frame


def accumulate_fromNetwork(event, frame):
    '''if normalize:
        # TODO edit normalization
        norm_factor = (ts1 + s * dt - e['timestamp']) / dt
    else:
        norm_factor = 1'''

    norm_factor = 1
    if event.polarity == 1:
        # event_frame[e['y'], e['x']] = (0, int(255 * norm_factor), 0)
        frame[event.y, event.x] = int(127 * norm_factor) + 127
    else:
        # event_frame[e['y'], e['x']] = (int(255 * norm_factor), 0, 0)
        frame[event.y, event.x] = 127 - int(127 * norm_factor)
    return frame