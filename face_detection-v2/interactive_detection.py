from argparse import ArgumentParser
from logging import getLogger, basicConfig, DEBUG, INFO, ERROR
import os
import sys
import detectors
import cv2
import math
import numpy as np
from timeit import default_timer as timer
from queue import Queue

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

FP32 = "extension/IR/FP32/"
FP16 = "extension/IR/FP16/"

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i","--input",help="Path to video file or image. 'cam' for capturing video stream from camera",
        required=True,
        type=str)
    parser.add_argument("-m_fc","--model_face",help="Optional. Path to an .xml file with a trained Age/Gender Recognition model.",
        type=str,
        default=None)
    parser.add_argument("-m_ag","--model_age_gender",help="Optional. Path to an .xml file with a trained Age/Gender Recognition model.",
        type=str,
        default=None)
    parser.add_argument("-m_em","--model_emotions",help="Optional. Path to an .xml file with a trained Emotions Recognition model.",
        type=str,
        default=None)
    parser.add_argument("-m_hp","--model_head_pose",
        help="Optional. Path to an .xml file with a trained Head Pose Estimation model.",
        type=str,
        default=None)
    parser.add_argument("-m_lm","--model_facial_landmarks",
        help="Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.",
        type=str,
        default=None)
    parser.add_argument("-l","--cpu_extension",
        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.",
        type=str,
        default=None)
    parser.add_argument("-d","--device",
        help="Specify the target device for MobileNet-SSSD / Face Detection to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default=None,
        choices=['CPU', 'GPU', 'HETERO:FPGA,CPU', 'MYRIAD'],
        type=str)
    parser.add_argument("-d_ag","--device_age_gender",
        help="Specify the target device for Age/Gender Recognition to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default=None,
        choices=['CPU', 'GPU', 'HETERO:FPGA,CPU', 'MYRIAD'],
        type=str)
    parser.add_argument("-d_em","--device_emotions",
        help="Specify the target device for for Emotions Recognition to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default=None,
        choices=['CPU', 'GPU', 'HETERO:FPGA,CPU', 'MYRIAD'],
        type=str)
    parser.add_argument("-d_hp","--device_head_pose", 
        help="Specify the target device for Head Pose Estimation to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default=None,
        choices=['CPU', 'GPU', 'HETERO:FPGA,CPU', 'MYRIAD'],
        type=str)
    parser.add_argument("-d_lm","--device_facial_landmarks",
        help="Specify the target device for Facial Landmarks Estimation to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default=None,
        choices=['CPU', 'GPU', 'HETERO:FPGA,CPU', 'MYRIAD'],
        type=str)
    parser.add_argument("-pp","--plugin_dir",
        help="Path to a plugin folder",
        type=str,
        default=None)
    parser.add_argument("-ptf","--prob_threshold_face",
        help="Probability threshold for face detections filtering",
        default=0.5,
        type=float)
    parser.add_argument("-o","--output_dir",
        help="Path to the output folder",
        type=str,
        default=None)
    return parser




class Detectors(object):
    def __init__(self, devices, models, cpu_extension, plugin_dir,
                 prob_threshold, prob_threshold_face, is_async_mode):
        self.cpu_extension = cpu_extension
        self.plugin_dir = plugin_dir
        self.prob_threshold = prob_threshold
        self.prob_threshold_face = prob_threshold_face
        self.is_async_mode = is_async_mode
        self._load_detectors(devices, models)

    def _load_detectors(self, devices, models):
        
        #Get the individual models and devices from their respective arrays
        
        device_fc= devices[0]
        device_ag= devices[1]
        device_em= devices[2]
        device_hp= devices[3]
        device_lm = devices[4]
        model_fc=models[0]
        model_ag=models[1]
        model_em=models[2]
        model_hp=models[3]
        model_lm=models[4]
        
        #Assign other default values
        
        cpu_extension = self.cpu_extension
        plugin_dir = self.plugin_dir
        prob_threshold = self.prob_threshold
        prob_threshold_face = self.prob_threshold_face
        is_async_mode = self.is_async_mode

        
        # Create face_detection class instance
        
        self.face_detectors = detectors.FaceDetection(
            device_fc, model_fc, cpu_extension, plugin_dir,
            prob_threshold_face, is_async_mode)
        
        # Create face_analytics class instances
        
        self.age_gender_detectors = detectors.AgeGenderDetection(
            device_ag, model_ag, cpu_extension, plugin_dir,
            prob_threshold_face, is_async_mode)
        self.emotions_detectors = detectors.EmotionsDetection(
            device_em, model_em, cpu_extension, plugin_dir,
            prob_threshold_face, is_async_mode)
        self.headpose_detectors = detectors.HeadPoseDetection(
            device_hp, model_hp, cpu_extension, plugin_dir, prob_threshold,
            is_async_mode)
        self.facial_landmarks_detectors = detectors.FacialLandmarksDetection(
            device_lm, model_lm, cpu_extension, plugin_dir,
            prob_threshold_face, is_async_mode)

    def _define_models(self, devices, models):
        
        device_fc= devices[0]
        device_ag= devices[1]
        device_em= devices[2]
        device_hp= devices[3]
        device_lm = devices[4]
        model_fc, model_ag, model_em, model_hp, model_lm = models

        # set devices and models
        fp_path = FP32 if device_fc == "CPU" else FP16
        model_fc = fp_path + model_fc_xml if model_fc is None else model_fc
        fp_path = FP32 if device_ag == "CPU" else FP16
        model_ag = fp_path + model_ag_xml if model_ag is None else model_ag
        fp_path = FP32 if device_em == "CPU" else FP16
        model_em = fp_path + model_em_xml if model_em is None else model_em
        fp_path = FP32 if device_hp == "CPU" else FP16
        model_hp = fp_path + model_hp_xml if model_hp is None else model_hp
        fp_path = FP32 if device_lm == "CPU" else FP16
        model_lm = fp_path + model_lm_xml if model_lm is None else model_lm

        return [model_ss, model_fc, model_ag, model_em, model_hp, model_lm]


class Detections(Detectors):
    def __init__(self, devices, models, cpu_extension, plugin_dir,
                 prob_threshold, prob_threshold_face, is_async_mode):
        super().__init__(devices, models, cpu_extension, plugin_dir,
                         prob_threshold, prob_threshold_face, is_async_mode)

        # initialize Calculate FPS
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()

    def face_detection(self, frame, next_frame, is_async_mode,
                       is_age_gender_detection, is_emotions_detection,
                       is_head_pose_detection, is_facial_landmarks_detection):

        # ----------- Start Face Detection ---------- #

        logger.debug("** face_detection start **")
        color = (0, 255, 0)
        det_time = 0
        det_time_ag = 0
        det_time_em = 0
        det_time_hp = 0
        det_time_lm = 0
        det_time_txt = ""

        frame_h, frame_w = frame.shape[:2]  # shape (h, w, c)
        is_face_analytics_enabled = True if is_age_gender_detection or is_emotions_detection else False

        inf_start = timer()
        self.face_detectors.submit_req(frame, next_frame, is_async_mode)
        ret = self.face_detectors.wait()
        faces = self.face_detectors.get_results(is_async_mode)
        inf_end = timer()
        det_time = inf_end - inf_start

        face_count = faces.shape[2]
        det_time_txt = "face_cnt:{} face:{:.3f} ms ".format(face_count, det_time * 1000)

        # ----------- Start Face Analytics ---------- #

        # Run face analytics with async mode when detected face count lager than 1.
        if face_count > 1:
            is_face_async_mode = True
        else:
            is_face_async_mode = False

        face_id = 0
        face_w, face_h = 0, 0
        face_frame = None
        next_face_frame = None
        prev_box = None

        face_q = Queue()
        for face in faces[0][0]:
            face_q.put(face)
        
        if is_face_async_mode:
            face_count = face_count + 1

        for face_id in range(face_count):
            face_id = 0
            face_analytics = ""
            age_gender = ""
            emotion = ""
            head_pose = ""
            facial_landmark = ""

            if not face_q.empty():
                face = face_q.get()

            box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            xmin, ymin, xmax, ymax = box.astype("int")
            class_id = int(face[1])
            result = str(face_id) + " " + str(round(face[2] * 100, 1)) + '% '

            if xmin < 0 or ymin < 0:
                logger.info(
                    "Rapid motion returns negative value(xmin and ymin) which make face_frame None. xmin:{} xmax:{} ymin:{} ymax:{}".
                    format(xmin, xmax, ymin, ymax))
                return frame

            # Start face analytics
            # prev_box is previous boxes(faces), which is None at the first time 
            # will be updated with prev face box in async mode
            if is_face_async_mode:
                next_face_frame = frame[ymin:ymax, xmin:xmax]
                if next_face_frame is None:
                    return frame
                if prev_box is not None:
                    xmin, ymin, xmax, ymax = prev_box.astype("int")
            else:
                face_frame = frame[ymin:ymax, xmin:xmax]

            # check face frame.
            # face_fame is None at the first time with async mode.
            if face_frame is not None:
                face_w, face_h = face_frame.shape[:2]
                # Resizing face_frame will be failed when witdh or height of the face_fame is 0 ex. (243, 0, 3)
                if face_w == 0 or face_h == 0:
                    logger.error(
                        "Unexpected shape of face frame. face_frame.shape:{} {}".
                        format(face_h, face_w))
                    return frame

            # ----------- Start Age/Gender detection ---------- #
            if is_age_gender_detection:
                logger.debug("*** age_gender_detection start ***")

                inf_start = timer()
                self.age_gender_detectors.submit_req(face_frame, next_face_frame, is_face_async_mode)
                ret = self.age_gender_detectors.wait()
                age, gender = self.age_gender_detectors.get_results(is_face_async_mode)
                age_gender = str(int(round(age))) + " " + gender + " "
                inf_end = timer()
                det_time_ag = inf_end - inf_start

                #det_time = det_time + det_time_ag
                det_time_ag += det_time_ag
                #det_time_txt = det_time_txt + "ag:{:.3f} ".format(det_time_ag * 1000)
                logger.debug("age:{} gender:{}".format(age, gender))
                logger.debug("*** age_gender_detection end ***")

            # ----------- Start Emotions detection ---------- #
            if is_emotions_detection:
                logger.debug("*** emotions detection start ***")

                inf_start = timer()
                self.emotions_detectors.submit_req(face_frame, next_face_frame, is_face_async_mode)
                ret = self.emotions_detectors.wait()
                emotion = self.emotions_detectors.get_results(is_face_async_mode)
                emotion = emotion + " "
                inf_end = timer()
                det_time_em = inf_end - inf_start

                #det_time = det_time + det_time_em
                det_time_em += det_time_em
                #det_time_txt = det_time_txt + "em:{:.3f} ".format(det_time_em * 1000)
                logger.debug("emotion:{}".format(emotion))
                logger.debug("*** emotion_detection end ***")

            # ----------- Start Head Pose detection ---------- #
            if is_head_pose_detection:
                logger.debug("*** head_pose_detection start ***")

                inf_start = timer()
                self.headpose_detectors.submit_req(face_frame, next_face_frame, is_face_async_mode)
                ret = self.headpose_detectors.wait()
                yaw, pitch, roll = self.headpose_detectors.get_results(is_face_async_mode)
                # face h/w will be 0 at the first inference with async mode
                if face_h != 0 and face_w != 0:
                    center_of_face = (xmin + face_h / 2, ymin + face_w / 2, 0)
                    frame = self.draw_axes(frame, center_of_face, yaw, pitch, roll, 50)
                inf_end = timer()
                det_time_hp = inf_end - inf_start

                
                det_time_hp += det_time_hp
                logger.debug("yaw(z):{:f}, pitch(y):{:f} roll(x):{:f}".format(yaw, pitch, roll))
                logger.debug("*** head_pose_detection end ***")

            # ----------- Start facial landmarks detection ---------- #
            if is_facial_landmarks_detection:
                logger.debug("*** landmarks_detection start ***")

                inf_start = timer()
                self.facial_landmarks_detectors.submit_req(face_frame, next_face_frame, is_face_async_mode)
                ret = self.facial_landmarks_detectors.wait()
                normed_landmarks = self.facial_landmarks_detectors.get_results(is_face_async_mode)
                n_lm = normed_landmarks.size
                for i in range(int(n_lm / 2)):
                    normed_x = normed_landmarks[2 * i]
                    normed_y = normed_landmarks[2 * i + 1]
                    x_lm = xmin + face_h * normed_x
                    y_lm = ymin + face_w * normed_y
                    cv2.circle(frame, (int(x_lm), int(y_lm)), 1 + int(0.012 * face_h), (0, 255, 255), -1)
                inf_end = timer()
                det_time_lm = inf_end - inf_start

                #det_time = det_time + det_time_lm
                det_time_lm += det_time_lm
                #det_time_txt = det_time_txt + "lm:{:.3f} ".format(det_time_lm * 1000)
                logger.debug("*** landmarks_detection end ***")

            face_id += 1

            if is_face_async_mode:
                face_frame = next_face_frame
                prev_box = box

            face_analytics = age_gender + emotion

            cv2.rectangle(frame, (xmin, ymin - 17), (xmax, ymin), color, -1)
            cv2.rectangle(frame, (xmin, ymin - 17), (xmax, ymin),
                          (255, 255, 255))
            # Draw box and label\class_id
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
            if is_face_analytics_enabled:
                cv2.putText(frame, face_analytics, (xmin + 3, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            else:
                cv2.putText(frame, result, (xmin + 3, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            logger.debug("face_id:{} confidence:{}%".format(
                face_id, round(face[2] * 100)))
        
        det_time = det_time + det_time_ag + det_time_em + det_time_hp + det_time_lm
        det_time_txt = det_time_txt + "ag:{:.3f} ".format(det_time_ag * 1000)
        det_time_txt = det_time_txt + "em:{:.3f} ".format(det_time_em * 1000)
        det_time_txt = det_time_txt + "hp:{:.3f} ".format(det_time_hp * 1000)
        det_time_txt = det_time_txt + "lm:{:.3f} ".format(det_time_lm * 1000)

        frame = self.draw_perf_stats(det_time, det_time_txt, frame,
                                     is_async_mode)

        return frame

    def draw_axes(self, frame, center_of_face, yaw, pitch, roll, scale):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0

        cx = int(center_of_face[0])
        cy = int(center_of_face[1])

        Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0],
                       [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx  # R = np.dot(Rz, np.dot(Ry, Rx))

        camera_matrix = self.build_camera_matrix(center_of_face, 950.0)

        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]

        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o

        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)

        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)

        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)

        return frame

    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix

    def draw_perf_stats(self, det_time, det_time_txt, frame, is_async_mode):

        # Draw FPS in top left corner
        fps = self.calc_fps()
        cv2.rectangle(frame, (frame.shape[1] - 50, 0), (frame.shape[1], 17),
                      (255, 255, 255), -1)
        cv2.putText(frame, fps, (frame.shape[1] - 50 + 3, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw performance stats
        inf_time_message = "Total Inference time: {:.3f} ms for async mode".format(det_time * 1000) if is_async_mode else \
            "Total Inference time: {:.3f} ms for sync mode".format(det_time * 1000)
        cv2.putText(frame, inf_time_message, (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 10, 10), 1)
        if det_time_txt:
            inf_time_message_each = "Detection time: {}".format(det_time_txt)
            cv2.putText(frame, inf_time_message_each, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 10, 10), 1)
        return frame

    def calc_fps(self):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps = self.curr_fps + 1

        if self.accum_time > 1:
            self.accum_time = self.accum_time - 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0

        return self.fps