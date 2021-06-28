import cv2
import os
from dv import AedatFile
import numpy as np


def check_dir(path):
    if not os.path.exists(path):
        print('Directory ' + '"if os.path.exists(path):"' + ' does not exists. /nCreating the dir...')
        os.makedirs(path)
    if not os.path.isdir(path):
        print('The path indicated is not a directory.')
        return False
    return True


def frame2avi(frames):
    """
        :param frames: List of Frames of the Aedat4 file.
        :return: Path of the created video.

        This function create a video (*.avi) from the Frames of the Aedat4 file.
    """
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


def only_video(file):
    with AedatFile(file) as f:

        # loop through the "frames" stream
        i = 0
        for frame in f['frames']:
            cv2.imshow('out', frame.image)
            cv2.waitKey(1)
            i += 1
        print(i)


def optical_flow(old_gray, frame_gray, features):
    mask = np.zeros_like(old_gray)
    lk_params = dict(winSize=(31, 31),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, features, None, **lk_params)
    # Select good points
    if p1 is not None:
        s = np.count_nonzero(st)
        good_new = np.empty((s, 2), np.float32)
        good_old = np.empty((s, 2), np.float32)
        i = 0
        k = 0
        for n in st:
            if n == 1:
                good_new[k] = p1[i]
                good_old[k] = features[i]
                k += 1
            i += 1

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)
        frame = cv2.circle(old_gray, (int(a), int(b)), 5, (0, 0, 0), -1)
    img = cv2.add(old_gray, mask)
    return img
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