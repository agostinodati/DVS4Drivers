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

        # loop through the "frames" stream
        i = 0
        for frame in f['frames']:
            ts = frame.timestamp

            cv2.imshow('out', frame.image)

            cv2.waitKey(1)
            i += 1
        print(i)

        events = np.hstack([packet for packet in f['events'].numpy()])

        normalize = True  # For normalization relative to timestamps
        k = 0
        s = 1
        time = 1000  # 100 fps
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
                    norm_factor = (ts+s*time - e['timestamp']) / time
                else:
                    norm_factor = 1

                if e['polarity'] == 1:
                    event_frame[e['y'], e['x']] = (0, int(255*norm_factor), 0)
                    # event_frame[e['y'], e['x']] = int(127 * norm_factor) + 127
                else:
                    event_frame[e['y'], e['x']] = (int(255*norm_factor), 0, 0)
                    # event_frame[e['y'], e['x']] = 127-int(127 * norm_factor)

            s += 1
            cv2.imshow('out3', event_frame)
            # Frame reset
            event_frame = np.zeros((height, width, 3), np.uint8)
            # event_frame[:, :, :] = 0
            cv2.waitKey(1)

        print(k)
        print(s)


if __name__ == '__main__':
    main()

