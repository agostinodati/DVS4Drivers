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
        for frame in f['frames']:
            ts = frame.timestamp

            cv2.imshow('out', frame.image)

            cv2.waitKey(1)
            i += 1
        print(i)

        k = 0
        s = 0
        time = 33000  # 30 fps
        ts = events[0]['timestamp']
        event_frame = np.zeros((height, width, 3), np.uint8)
        while k != len(events):

            for j in range(k, len(events)):
                e = events[j]
                k += 1

                if e['polarity'] == 1:
                    event_frame[e['y'], e['x']] = (0, 255, 0)
                else:
                    event_frame[e['y'], e['x']] = (0, 0, 255)

                if k == len(events) or e['timestamp'] != events[j + 1]['timestamp']:
                    break

            # 33 millisecond skip for each frame (30 fps video)
            # All events in this time interval are combined into one frame
            if k == len(events) or events[k]['timestamp'] < ts + s*time:
                continue

            s += 1
            cv2.imshow('out2', event_frame)
            # Frame reset
            event_frame = np.zeros((height, width, 3), np.uint8)
            cv2.waitKey(1)

        print(k)
        print(s)


if __name__ == '__main__':
    main()

