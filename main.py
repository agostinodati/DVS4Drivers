from dv import NetworkEventInput
from dv import AedatFile
import cv2


def main():
    with AedatFile("C:/Users/User/Downloads/dvSave-2021_04_23_13_45_03.aedat4") as f:
        # list all the names of streams in the file
        print(f.names)

        # Access dimensions of the event stream
        height, width = f['events'].size

        '''# loop through the "events" stream
        for e in f['events']:
            print(e.timestamp)'''

        # loop through the "events" stream as numpy packets
        for e in f['events'].numpy():
            print(e.shape)

        # loop through the "frames" stream
        for frame in f['frames']:
            print(frame.timestamp)
            cv2.imshow('out', frame.image)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()

