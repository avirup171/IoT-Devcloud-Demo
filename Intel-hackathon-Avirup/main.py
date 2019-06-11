from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
import numpy as np
import io
import detect as dt
from openvino.inference_engine import IENetwork, IEPlugin
from pathlib import Path
import json
sys.path.insert(0, str(Path().resolve().parent.parent))
from demoTools.demoutils import progressUpdate
import detect

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera",
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA, MYRIAD or HDDL is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
                        default=None, type=str)
    return parser


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        pass

 
def main():
    job_id = os.environ['PBS_JOBID']
    args = build_argparser().parse_args()
    model_xml=args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    device=args.device
    cpu_extension=args.cpu_extension
    plugin_dir=args.plugin_dir
    device=args.device
    is_async_mode = True
    object_detection=dt.Detectors(device,args.model,cpu_extension,plugin_dir,is_async_mode)
    resultant_initialisation_object=object_detection.initialise_inference()
    input_stream = args.input
    base_dir = os.getcwd()
    result_file = open(os.path.join(args.output_dir,'output_'+str(job_id)+'.txt'), "w")
    progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')
    render_file_path = os.path.join(args.output_dir,'r_progress_'+str(job_id)+'.txt')
    #Start video capturing process
    cap = cv2.VideoCapture(input_stream)
    cap_r = cv2.VideoCapture(input_stream)
    print(input_stream)
    #Frame count
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_len)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    cur_request_id = 0
    next_request_id = 1
    out = cv2.VideoWriter(args.output_dir+job_id+".mp4",0x00000021, 50.0, (frame_width,frame_height))
    print("Entered")
    begin_time=time.time()
    res_array=[]
    frame_count=1
    try:
        while cap.isOpened():
            print("Entered 1")
            ret, frame = cap.read()
            if not ret:
                break
            initial_w = cap.get(3)
            initial_h = cap.get(4)
            res_inference=resultant_initialisation_object.process_frame(cur_request_id,next_request_id,frame,initial_h,initial_w,False)
            res_array.append(res_inference)
            if frame_count%10 == 0: 
                progressUpdate(progress_file_path, time.time()-begin_time, frame_count, video_len)
            frame_count+=1
        cap.release()
    finally:
        del resultant_initialisation_object.exec_net
    i=0
    render_time_start=time.time()
    frame_count=1
    while cap_r.isOpened():
        print("Rendering ...")
        ret, frame = cap_r.read()
        if not ret:
            break
        initial_w = cap_r.get(3)
        initial_h = cap_r.get(4)
        resultant_frame_processed=resultant_initialisation_object.placeBoxes(res_array[i],None,0.4,frame,initial_w,initial_h,False,cur_request_id)
        i+=1
        frame_count+=1
        if frame_count%10 == 0: 
            progressUpdate(render_file_path, time.time()-render_time_start, frame_count, len(resultant_frame_processed))
        out.write(resultant_frame_processed)
    out.release()
    total_time = time.time() - begin_time
    with open(os.path.join(args.output_dir, 'stats_'+str(job_id)+'.txt'), 'w') as f:
        f.write(str(round(total_time, 1))+'\n')
        f.write(str(frame_count)+'\n')
    #resultant_frame=resultant_initialisation_object.placeBoxes(res_inference,None,0.1,frame,initial_w,initial_h,False,cur_request_id)

if __name__ == '__main__':
    sys.exit(main() or 0)
