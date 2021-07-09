import numpy.distutils.command.build_src
from dv import AedatFile
import cv2
import numpy as np
import matplotlib.pyplot as pl
import dvs4d_lib

amal1 = "C:/Users/User/Downloads/dvSave-2021_04_23_13_45_03.aedat4"
amal2 = "D:/Utorrent/dvSave-2021_05_28_18_48_58.aedat4"
amal3 = "C:/Users/User/Downloads/dvSave-2021_06_28_23_03_03.aedat4"
ago1 = "D:/Download/mancini.aedat4"
ago2 = "D:/Download/mancini_notte.aedat4"
output = "errorlog"


def main_optical_flow_naive(timeskip=0):
    '''

    Calculate the optical flow using a naive event visualizer method.
    '''
    with AedatFile(ago1) as f:

        # Access dimensions of the event stream
        height, width = f['events'].size

        # Time-skip for the aedat file ---
        video_frame = f['frames'].__next__()
        fast_forward = video_frame.timestamp + timeskip * 1000000

        for packet in f['events'].numpy():
            e = packet[-1]
            while video_frame.timestamp <= e['timestamp']:
                video_frame = f['frames'].__next__()
            if e['timestamp'] > fast_forward:
                break;
        # ---

        normalize = False  # For normalization relative to timestamps

        event_dt = 8000  # Temporal window (micro-seconds) for event frames (for 100 fps -> 10000 us)
        video_dt = 39980  # Approximated micro-seconds of the video frames
        attenuation_factor = 32  # Factor used in the decrement of events drawn (accumulator)
        delay_old_frame = 0  # Delay before starting to render the old event frame
        advance_new_frame = 0  # Advance before starting to render the new event frame
        accum_dt = 10000  # Temporal window before starting decrement the accumulator frame
        accum_ref_ts = 0  # Timestamp of the previous accumulator frame
        accum_increment = 30  # The increment of intensity of the accumulator when an event occurs

        isolate_face_roi = False  # Check variable to isolate (or not) the ROI of the face

        missing_frames_amt = 10  # Amount of frame to skip to simulate a test case

        count_accumulator = 0
        count_video = 0

        errors = []  # List where will be stored the errors of the optical flow at every iteration

        new_event_frame = np.zeros((height, width, 1), np.uint8)  # The event frame closest to the current video frame
        accumulator_frame = new_event_frame.copy()
        new_event_frame[:, :, 0] = 127  # Initialize to gray the image
        old_event_frame = new_event_frame.copy()  # The event frame closest to the previous video frame

        # Access to a video frame to initialize the needed parameters:
        # - old_landmarks: landmarks calculated using facemesh on the videoframe
        video_frame = f['frames'].__next__()
        old_landmarks, is_video = dvs4d_lib.find_landmarks(video_frame.image, video_frame.image)
        prev_videoframe_ts = video_frame.timestamp
        prev_facemesh_fail = False  # Flag used if facemesh failed during the previous video frame
        fail_counter = 0

        # Iterate through the events of the Aedat file
        for packet in f['events'].numpy():
            for e in packet.tolist():
                ts = e[0]  # Event's timestamp

                # If the timestamp "ts" is in this allowed temporal window, draw the old event frame
                if prev_videoframe_ts + delay_old_frame <= ts < prev_videoframe_ts + delay_old_frame + event_dt:
                    old_event_frame = dvs4d_lib.naive_event_drawer(normalize, e, old_event_frame, event_dt,
                                                                 prev_videoframe_ts + delay_old_frame + event_dt)

                # If the timestamp "ts" is in this allowed temporal window, draw the new event frame
                if prev_videoframe_ts + video_dt - advance_new_frame - event_dt <= ts < prev_videoframe_ts + video_dt - advance_new_frame:
                    new_event_frame = dvs4d_lib.naive_event_drawer(normalize, e, new_event_frame, event_dt,
                                                                 prev_videoframe_ts + video_dt - advance_new_frame)

                # Draw the accumulator frame and eventually decrement the intensity values
                accumulator_frame = dvs4d_lib.accumulator(e, accumulator_frame, accum_increment)
                if ts > accum_ref_ts + accum_dt:
                    accumulator_frame = numpy.subtract(accumulator_frame, accumulator_frame / attenuation_factor)
                    accumulator_frame = accumulator_frame.astype(np.uint8)
                    accum_ref_ts = ts

                # Synchronize event ad video frames
                if ts >= prev_videoframe_ts + video_dt:
                    while video_frame.timestamp <= ts:
                        try:
                            video_frame = f['frames'].__next__()
                        except:
                            break

                        prev_videoframe_ts = video_frame.timestamp

                        if isolate_face_roi:
                            old_event_frame, new_event_frame = dvs4d_lib.face_roi(old_landmarks, old_event_frame,
                                                                                  new_event_frame)
                        cv2.imshow('Video', video_frame.image)
                        cv2.imshow('Accumulator', accumulator_frame)
                        cv2.imshow('Old Event Frame', old_event_frame)
                        cv2.imshow('New Event Frame', new_event_frame)

                        # Calculate new landmarks
                        new_landmarks_true, is_video = dvs4d_lib.find_landmarks_only_video(video_frame.image)

                        # Skip n-frames to simulate a situation of test
                        fail_counter += 1
                        if fail_counter <= missing_frames_amt:
                            new_landmarks = None
                        else:
                            fail_counter = 0
                            new_landmarks = new_landmarks_true

                        # If facemesh didn't find the landmarks then they will be predicted using optical flow
                        facemesh_fail = False
                        if new_landmarks is None and old_landmarks is not None:
                            facemesh_fail = True
                            if facemesh_fail and prev_facemesh_fail:
                                new_landmarks = dvs4d_lib.optical_flow(prev_stored_new_frame, new_event_frame,
                                                                       old_landmarks)
                            else:
                                new_landmarks = dvs4d_lib.optical_flow(old_event_frame, new_event_frame,
                                                                       old_landmarks)
                            if is_video:
                                to_draw = video_frame.image
                                if new_landmarks_true is not None:
                                    count_video += 1
                            else:
                                to_draw = accumulator_frame
                                count_accumulator += 1

                            error = dvs4d_lib.draw_landmarks_optical_flow(old_landmarks, new_landmarks, to_draw,
                                                                          new_landmarks_true)
                            if error is not None:
                                errors.append(error)
                        old_landmarks = new_landmarks

                    prev_facemesh_fail = facemesh_fail
                    prev_stored_new_frame = new_event_frame.copy()
                    # Frame reset for events draw
                    new_event_frame[:, :, 0] = 127
                    old_event_frame[:, :, 0] = 127
                    cv2.waitKey(1)
                    print("Accumulator count: " + str(count_accumulator))
                    print("Video count: " + str(count_video))

        # Logging the error
        error_np = np.asarray(errors)
        avg_error = np.average(error_np)
        np.save(output, error_np, allow_pickle=False)
        pl.plot(error_np)
        pl.axhline(y=avg_error, color='r')
        pl.savefig(
            'Naive_errorplot-DT_' + str(event_dt) + '-INCREMENT_' + str(accum_increment) + '-ACCUMTS_' + str(accum_dt))
        pl.show()


def main_optical_flow_accumulator(timeskip=0):
    '''

    Calculate the optical flow using an accumulator of events.
    '''
    with AedatFile(ago1) as f:

        # Access dimensions of the event stream
        height, width = f['events'].size

        # Time-skip for the aedat file ---
        video_frame = f['frames'].__next__()
        fast_forward = video_frame.timestamp + timeskip * 1000000

        for packet in f['events'].numpy():
            e = packet[-1]
            while video_frame.timestamp <= e['timestamp']:
                video_frame = f['frames'].__next__()
            if e['timestamp'] > fast_forward:
                break;
        # ---

        normalize = False  # For normalization relative to timestamps

        video_dt = 39980  # Approximated micro-seconds of the video frames
        attenuation_factor = 32  # Factor used in the decrement of events drawn (accumulator)
        delay_old_frame = 0  # Delay before starting to render the old event frame
        advance_new_frame = 0  # Advance before starting to render the new event frame
        accum_dt = 10000  # Temporal window before starting decrement the accumulator frame
        accum_ref_ts = 0  # Timestamp of the previous accumulator frame
        accum_increment = 30  # The increment of intensity of the accumulator when an event occurs

        isolate_face_roi = True  # Check variable to isolate (or not) the ROI of the face

        missing_frames_amt = 10  # Amount of frame to skip to simulate a test case

        count_accumulator = 0
        count_video = 0

        errors = []  # List where will be stored the errors of the optical flow at every iteration

        new_event_frame = np.zeros((height, width, 1), np.uint8)  # The event frame closest to the current video frame
        accumulator_frame = new_event_frame.copy()
        new_event_frame[:, :, 0] = 127  # Initialize to gray the image
        old_event_frame = new_event_frame.copy()  # The event frame closest to the previous video frame

        # Access to a video frame to initialize the needed parameters:
        # - old_landmarks: landmarks calculated using facemesh on the videoframe
        video_frame = f['frames'].__next__()
        old_landmarks, is_video = dvs4d_lib.find_landmarks(video_frame.image, video_frame.image)
        prev_videoframe_ts = video_frame.timestamp
        prev_facemesh_fail = False  # Flag used if facemesh failed during the previous video frame
        fail_counter = 0

        # Iterate through the events of the Aedat file
        for packet in f['events'].numpy():
            for e in packet.tolist():
                ts = e[0]  # Event's timestamp

                # Make a copy of the accumulator frame at different time
                accumulator_frame = dvs4d_lib.accumulator(e, accumulator_frame, accum_increment)
                if prev_videoframe_ts + delay_old_frame <= ts:
                    old_event_frame = accumulator_frame.copy()
                if ts < prev_videoframe_ts + video_dt - advance_new_frame:
                    new_event_frame = accumulator_frame.copy()

                # Draw the accumulator frame and eventually decrement the intensity values
                accumulator_frame = dvs4d_lib.accumulator(e, accumulator_frame, accum_increment)
                if ts > accum_ref_ts + accum_dt:
                    accumulator_frame = numpy.subtract(accumulator_frame, accumulator_frame / attenuation_factor)
                    accumulator_frame = accumulator_frame.astype(np.uint8)
                    accum_ref_ts = ts

                # Synchronize event ad video frames
                if ts >= prev_videoframe_ts + video_dt:
                    while video_frame.timestamp <= ts:
                        try:
                            video_frame = f['frames'].__next__()
                        except:
                            break

                        prev_videoframe_ts = video_frame.timestamp

                        if isolate_face_roi:
                            old_event_frame, new_event_frame = dvs4d_lib.face_roi(old_landmarks, old_event_frame,
                                                                                  new_event_frame)
                        cv2.imshow('Video', video_frame.image)
                        cv2.imshow('Accumulator', accumulator_frame)
                        cv2.imshow('Old Event Frame', old_event_frame)
                        cv2.imshow('New Event Frame', new_event_frame)

                        # Calculate new landmarks
                        new_landmarks_true, is_video = dvs4d_lib.find_landmarks_only_video(video_frame.image)

                        # Skip n-frames to simulate a situation of test
                        fail_counter += 1
                        if fail_counter <= missing_frames_amt:
                            new_landmarks = None
                        else:
                            fail_counter = 0
                            new_landmarks = new_landmarks_true

                        # If facemesh didn't find the landmarks then they will be predicted using optical flow
                        facemesh_fail = False
                        if new_landmarks is None and old_landmarks is not None:
                            facemesh_fail = True
                            if facemesh_fail and prev_facemesh_fail:
                                new_landmarks = dvs4d_lib.optical_flow(prev_stored_new_frame, new_event_frame,
                                                                       old_landmarks)
                            else:
                                new_landmarks = dvs4d_lib.optical_flow(old_event_frame, new_event_frame,
                                                                       old_landmarks)
                            if is_video:
                                to_draw = video_frame.image
                                if new_landmarks_true is not None:
                                    count_video += 1
                            else:
                                to_draw = accumulator_frame
                                count_accumulator += 1

                            error = dvs4d_lib.draw_landmarks_optical_flow(old_landmarks, new_landmarks, to_draw,
                                                                          new_landmarks_true)
                            if error is not None:
                                errors.append(error)
                        old_landmarks = new_landmarks

                    prev_facemesh_fail = facemesh_fail
                    prev_stored_new_frame = new_event_frame.copy()

                    cv2.waitKey(1)
                    print("Accumulator count: " + str(count_accumulator))
                    print("Video count: " + str(count_video))

        # Logging the error
        error_np = np.asarray(errors)
        avg_error = np.average(error_np)
        np.save(output, error_np, allow_pickle=False)
        pl.plot(error_np)
        pl.axhline(y=avg_error, color='r')
        pl.savefig(
            'Accumulator_errorplot' + '-INCREMENT_' + str(accum_increment) + '-ACCUMTS_' + str(accum_dt))
        pl.show()


if __name__ == '__main__':
    main_optical_flow_accumulator()
