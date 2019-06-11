from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2 as cv
import time
import logging as log
import numpy as np
import io
from openvino.inference_engine import IENetwork, IEPlugin
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
from demoTools.demoutils import progressUpdate
import interactive_detection


def main():
    
    args = interactive_detection.build_argparser().parse_args()
    
    device_age_gender= device_emotions =device_head_pose = device_facial_landmarks = None
    
    #Args device handler
    if args.device_age_gender is None:
        device_age_gender=args.device
        print(device_age_gender)
    if args.device_emotions is None:
        device_emotions=args.device
        print(device_emotions)
    if args.device_head_pose is None:
        device_head_pose=args.device
        print(device_head_pose)
    if args.device_head_pose is None:
        device_facial_landmarks=args.device
        print(device_facial_landmarks)
    
    devices = [
        args.device, args.device, device_age_gender, device_emotions,
        device_head_pose, device_facial_landmarks
    ]
    
    models = [
        args.model_face, args.model_age_gender,
        args.model_emotions, args.model_head_pose, args.model_facial_landmarks
    ]
    
    
    if "CPU" in devices and args.cpu_extension is None:
        print(
            "\nPlease try to specify cpu extensions library path in demo's command line parameters using -l "
            "or --cpu_extension command line argument")
        sys.exit(1)
    
    job_id = os.environ['PBS_JOBID']
    #Resultant and progress file path
    base_dir = os.getcwd()
    result_file = open(os.path.join(args.output_dir,'output_'+str(job_id)+'.txt'), "w")
    progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')
    render_file_path = os.path.join(args.output_dir,'r_progress_'+str(job_id)+'.txt')
    frame_count=0
    is_async_mode=False
    is_age_gender_detection=True
    is_emotions_detection=True
    is_head_pose_detection=True
    is_facial_landmarks_detection=True
    # Create detectors class instance
    detections = interactive_detection.Detections(
        devices, models, args.cpu_extension, args.plugin_dir,
        args.prob_threshold_face, args.prob_threshold_face, is_async_mode)
    
    cap = cv.VideoCapture(args.input)
    #Frame count
    video_len = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(args.input)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv.VideoWriter(args.output_dir+job_id+".mp4",0x00000021, 50.0, (frame_width,frame_height))
    begin_time=time.time()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_array=[]
    while cap.isOpened():
        infer_time_start=time.time()
        ret, frame = cap.read()
        if ret == True:
            resultant_frame= detections.face_detection(
                    frame, frame, is_async_mode, is_age_gender_detection,
                    is_emotions_detection, is_head_pose_detection,is_facial_landmarks_detection)
            frame_count+=1
            #Write data to progress tracker
            if frame_count%10 == 0: 
                progressUpdate(progress_file_path, time.time()-begin_time, frame_count, video_len)
            frame_array.append(resultant_frame)
        else:
            break
    # When everything done, release the video capture and video write objects
    cap.release()
    #out.release()
    total_time = time.time() - begin_time
    with open(os.path.join(args.output_dir, 'stats_'+str(job_id)+'.txt'), 'w') as f:
        f.write(str(round(total_time, 1))+'\n')
        f.write(str(frame_count)+'\n')
        
    frame_count=0
    render_time_start=time.time()
    for i in range(len(frame_array)):
        frame_count+=1
        if frame_count%10 == 0: 
            progressUpdate(render_file_path, time.time()-render_time_start, frame_count, len(frame_array))
        out.write(frame_array[i])
    out.release()
    
    ellapsed_time=time.time()-begin_time
    time_results_dir= os.path.join(base_dir,'time_results_dir')
    make_sure_path_exists(time_results_dir)
    with open(os.path.join(time_results_dir, 'stats_'+str(job_id)+'.txt'), 'w') as f:
        f.write(str(round(ellapsed_time, 1))+'\n')
        f.write(str(frame_count)+'\n')
    
    # Closes all the frames
    cv.destroyAllWindows()
    
    


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        pass




if __name__ == '__main__':
    sys.exit(main() or 0)
