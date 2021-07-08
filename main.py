import numpy.distutils.command.build_src
from dv import AedatFile
import cv2
import numpy as np
import utility
import random
import matplotlib.pyplot as pl
import dvs4d_lib

amal1 = "C:/Users/User/Downloads/dvSave-2021_04_23_13_45_03.aedat4"
amal2 = "D:/Utorrent/dvSave-2021_05_28_18_48_58.aedat4"
amal3 = "C:/Users/User/Downloads/dvSave-2021_06_28_23_03_03.aedat4"
ago1 = "D:/Download/mancini.aedat4"
ago2 = "D:/Download/mancini_notte.aedat4"
output = "errorlog"


def main_optical_flow_naive():
    with AedatFile(ago1) as f:

        # Access dimensions of the event stream
        height, width = f['events'].size

        video_frame = f['frames'].__next__()
        fast_forward = video_frame.timestamp + 0 * 1000000

        for packet in f['events'].numpy():
            e = packet[-1]
            while video_frame.timestamp <= e['timestamp']:
                video_frame = f['frames'].__next__()
            if e['timestamp'] > fast_forward:
                break;

        normalize = False  # For normalization relative to timestamps

        event_dt = 8000  # for 100 fps -> 10000 us
        video_dt = 39980
        attenuation_factor = 32
        delay_old_frame = 0
        advance_new_frame = 0
        accum_dt = 10000
        accum_ref_ts = 0
        accum_increment = 30

        isolate_face_roi = False

        missing_frames_amt = 10
        count_accumulator = 0
        count_video = 0

        errors = []

        new_event_frame = np.zeros((height, width, 1), np.uint8)
        accumulator_frame = new_event_frame.copy()
        new_event_frame[:, :, 0] = 127
        old_event_frame = new_event_frame.copy()

        video_frame = f['frames'].__next__()
        old_landmarks, is_video = dvs4d_lib.find_landmarks(video_frame.image, video_frame.image)
        prev_videoframe_ts = video_frame.timestamp
        prev_facemesh_fail = False
        fail_counter = 0
        for packet in f['events'].numpy():
            for e in packet.tolist():
                ts = e[0]

                if prev_videoframe_ts + delay_old_frame <= ts < prev_videoframe_ts + delay_old_frame + event_dt:
                    old_event_frame = utility.naive_event_drawer(normalize, e, old_event_frame, event_dt,
                                                                 prev_videoframe_ts + delay_old_frame + event_dt)

                if prev_videoframe_ts + video_dt - advance_new_frame - event_dt <= ts < prev_videoframe_ts + video_dt - advance_new_frame:
                    new_event_frame = utility.naive_event_drawer(normalize, e, new_event_frame, event_dt,
                                                                 prev_videoframe_ts + video_dt - advance_new_frame)

                accumulator_frame = utility.accumulator(e, accumulator_frame, accum_increment)

                if ts > accum_ref_ts + accum_dt:
                    accumulator_frame = numpy.subtract(accumulator_frame, accumulator_frame / attenuation_factor)
                    accumulator_frame = accumulator_frame.astype(np.uint8)
                    accum_ref_ts = ts

                # 1 millisecond skip for each frame (100 fps video)
                # All events in this time window are combined into one frame
                if ts >= prev_videoframe_ts + video_dt:
                    while video_frame.timestamp <= ts:
                        try:
                            video_frame = f['frames'].__next__()
                        except:
                            break
                        prev_videoframe_ts = video_frame.timestamp

                        if isolate_face_roi:
                            old_event_frame, new_event_frame = utility.face_roi(old_landmarks, old_event_frame,
                                                                                new_event_frame)
                        cv2.imshow('Video', video_frame.image)
                        cv2.imshow('Accumulator', accumulator_frame)
                        cv2.imshow('Old Event Frame', old_event_frame)
                        cv2.imshow('New Event Frame', new_event_frame)

                        new_landmarks_true, is_video = dvs4d_lib.find_landmarks_only_video(video_frame.image)

                        fail_counter += 1
                        if fail_counter <= missing_frames_amt:
                            new_landmarks = None
                        else:
                            fail_counter = 0
                            new_landmarks = new_landmarks_true

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
                    # Frame reset
                    new_event_frame[:, :, 0] = 127
                    old_event_frame[:, :, 0] = 127
                    cv2.waitKey(1)
                    print("Accumulator count: " + str(count_accumulator))
                    print("Video count: " + str(count_video))

        error_np = np.asarray(errors)
        avg_error = np.average(error_np)
        np.save(output, error_np, allow_pickle=False)
        pl.plot(error_np)
        pl.axhline(y=avg_error, color='r')
        pl.savefig('Naive_errorplot-DT_' + str(event_dt) + '-INCREMENT_' + str(accum_increment) + '-ACCUMTS_' + str(accum_dt))
        pl.show()


def main_optical_flow_accumulator():
    with AedatFile(ago1) as f:

        # Access dimensions of the event stream
        height, width = f['events'].size

        video_frame = f['frames'].__next__()
        fast_forward = video_frame.timestamp + 0 * 1000000

        for packet in f['events'].numpy():
            e = packet[-1]
            while video_frame.timestamp <= e['timestamp']:
                video_frame = f['frames'].__next__()
            if e['timestamp'] > fast_forward:
                break;

        normalize = False  # For normalization relative to timestamps

        video_dt = 39980
        attenuation_factor = 32
        delay_old_frame = 0
        advance_new_frame = 0
        accum_dt = 10000
        accum_ref_ts = 0
        accum_increment = 30

        isolate_face_roi = False

        missing_frames_amt = 10
        count_accumulator = 0
        count_video = 0

        errors = []

        new_event_frame = np.zeros((height, width, 1), np.uint8)
        accumulator_frame = new_event_frame.copy()
        new_event_frame[:, :, 0] = 127
        old_event_frame = new_event_frame.copy()

        video_frame = f['frames'].__next__()
        old_landmarks, is_video = dvs4d_lib.find_landmarks(video_frame.image, video_frame.image)
        prev_videoframe_ts = video_frame.timestamp
        prev_facemesh_fail = False
        fail_counter = 0
        for packet in f['events'].numpy():
            for e in packet.tolist():
                ts = e[0]

                accumulator_frame = utility.accumulator(e, accumulator_frame, accum_increment)
                if prev_videoframe_ts + delay_old_frame <= ts:
                    old_event_frame = accumulator_frame.copy()
                if ts < prev_videoframe_ts + video_dt - advance_new_frame:
                    new_event_frame = accumulator_frame.copy()

                if ts > accum_ref_ts + accum_dt:
                    accumulator_frame = numpy.subtract(accumulator_frame, accumulator_frame / attenuation_factor)
                    accumulator_frame = accumulator_frame.astype(np.uint8)
                    accum_ref_ts = ts

                # 1 millisecond skip for each frame (100 fps video)
                # All events in this time window are combined into one frame
                if ts >= prev_videoframe_ts + video_dt:
                    while video_frame.timestamp <= ts:
                        try:
                            video_frame = f['frames'].__next__()
                        except:
                            break
                        prev_videoframe_ts = video_frame.timestamp

                        if isolate_face_roi:
                            old_event_frame, new_event_frame = utility.face_roi(old_landmarks, old_event_frame,
                                                                                new_event_frame)
                        cv2.imshow('Video', video_frame.image)
                        cv2.imshow('Accumulator', accumulator_frame)
                        cv2.imshow('Old Event Frame', old_event_frame)
                        cv2.imshow('New Event Frame', new_event_frame)

                        new_landmarks_true, is_video = dvs4d_lib.find_landmarks_only_video(video_frame.image)

                        fail_counter += 1
                        if fail_counter <= missing_frames_amt:
                            new_landmarks = None
                        else:
                            fail_counter = 0
                            new_landmarks = new_landmarks_true

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

        error_np = np.asarray(errors)
        avg_error = np.average(error_np)
        np.save(output, error_np, allow_pickle=False)
        pl.plot(error_np)
        pl.axhline(y=avg_error, color='r')
        pl.savefig('Accumulator_errorplot' + '-INCREMENT_' + str(accum_increment) + '-ACCUMTS_' + str(accum_dt))
        pl.show()


if __name__ == '__main__':
    main_optical_flow_accumulator()
