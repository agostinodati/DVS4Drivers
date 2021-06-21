import cv2
import os


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
