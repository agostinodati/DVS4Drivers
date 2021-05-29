from dv import NetworkEventInput
from dv import AedatFile
import cv2
import numpy as np


def main():
    with AedatFile("C:/Users/User/Downloads/dvSave-2021_04_23_13_45_03.aedat4") as f:
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

        events = np.hstack([packet for packet in f['events'].numpy()])

        # loop through the "frames" stream
        i = 0
        k = 0
        event_frame = np.zeros((height, width, 3), np.uint8)
        for frame in f['frames']:
            ts = frame.timestamp
            for e in events[k:]:
                if e['polarity'] == 1:
                    event_frame[e['y'], e['x']] = (0, 255, 0)
                else:
                    event_frame[e['y'], e['x']] = (255, 0, 0)
                k += 1
                if e['timestamp'] != ts:
                    break
            cv2.imshow('out', frame.image)
            cv2.imshow('out2', event_frame)
            cv2.waitKey(0)
            i += 1
            print(k)
        print(i)


if __name__ == '__main__':
    main()

