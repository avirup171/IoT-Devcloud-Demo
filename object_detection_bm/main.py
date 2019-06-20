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
sys.path.insert(0, str(Path().resolve().parent.parent))
from demoTools.demoutils import progressUpdate
from datetime import datetime



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
    parser.add_argument('-nireq', '--number_infer_requests', type=int, required=False, default=8,
                        help="Number of inference requests")
    parser.add_argument('-niter', '--number_iterations', type=int, required=False, default=8,
                        help="Number of iterations requests")
    return parser

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        pass

def main():
    res_frame_array=[]
    job_id = os.environ['PBS_JOBID']
    is_async_mode = True
    args = build_argparser().parse_args()
    number_infer_requests=args.number_infer_requests
    model_xml=args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    base_dir = os.getcwd()
    result_file = open(os.path.join(args.output_dir,'output_'+str(job_id)+'.txt'), "w")
    progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')
    render_file_path = os.path.join(args.output_dir,'r_progress_'+str(job_id)+'.txt')
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading plugins for {} device...".format(args.device))
        plugin.add_cpu_extension(args.cpu_extension)
        
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
            format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
            sys.exit(1)
        

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
        
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exe_network = plugin.load(network=net, num_requests=args.number_infer_requests)
    if isinstance(net.inputs[input_blob], list):
        n, c, h, w = net.inputs[input_blob]
    else:
        n, c, h, w = net.inputs[input_blob].shape
    del net
    


    input_stream = args.input
    #Start video capturing process
    if args.input==None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_stream)
        cap_r = cv2.VideoCapture(input_stream)
    #Frame count
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(args.output_dir+job_id+".mp4",0x00000021, 50.0, (frame_width,frame_height))
    begin_time=time.time()
    
    required_inference_requests_were_executed = False
    res_array=[]
    frame_array=[]
    res_array=[]
    frame_count=0
    not_completed_index=0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            initial_w = cap.get(3)
            initial_h = cap.get(4)
            frame_count+=1
            #if args.number_iterations is not None:
            #    steps_count += args.number_iterations
            #Preprocessing
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h,w))
                        
            frame_array_temp=[None]*args.number_infer_requests
            start_time = datetime.now()
            frame_array.append(in_frame)
            if (len(frame_array)-1==(args.number_infer_requests)):
                print("Initiate")   
                for i in range(args.number_infer_requests):
                    exe_network.start_async(request_id=i, inputs={input_blob: frame_array[i]})
                print("results")
                
                while not required_inference_requests_were_executed and exe_network.requests[not_completed_index].wait() is 0:
                    if not_completed_index == (args.number_infer_requests-1):
                        required_inference_requests_were_executed=True
                    res = exe_network.requests[not_completed_index].outputs[out_blob]
                    frame_array_temp[not_completed_index]=res
                    not_completed_index+=1
                not_completed_index=0
                required_inference_requests_were_executed=False
                res_array.append(frame_array_temp)
                frame_array=[]
            
            if frame_count%10 == 0: 
                progressUpdate(progress_file_path, time.time()-begin_time, frame_count, video_len)
            res=None
            step=0
            key = cv2.waitKey(1)
            if key == 27:
                break
        #out.release()
        cap.release()
    finally:
        del exe_network
        del plugin
    
    total_time = time.time() - begin_time
    with open(os.path.join(args.output_dir, 'stats_'+str(job_id)+'.txt'), 'w') as f:
        f.write(str(round(total_time, 1))+'\n')
        f.write(str(frame_count)+'\n')
    frame_count=0
    render_time_start=time.time()
    resultant_array_indexer=0
    print("Rendering ...")
    resultant_linear_array=[]
    for i in range(len(res_array)):
        temp_res_array=res_array[i]
        for j in range(len(temp_res_array)):
            resultant_linear_array.append(temp_res_array[j])
    i=0
    while cap_r.isOpened():
        print("Rendering ...")
        ret, frame = cap_r.read()
        if not ret:
            break
        initial_w = cap_r.get(3)
        initial_h = cap_r.get(4)
        resultant_frame_processed=placeBoxes(resultant_linear_array[i],None,0.5,frame,initial_w,initial_h,True)
        i+=1
        frame_count+=1
        if frame_count%10 == 0: 
            progressUpdate(render_file_path, time.time()-render_time_start, frame_count, len(resultant_frame_processed))
        out.write(resultant_frame_processed)
    out.release()
    
def placeBoxes(res, labels_map, prob_threshold, frame, initial_w, initial_h, is_async_mode):
        for obj in res[0][0]:
            # Draw only objects when probability more than specified threshold
            if obj[2] > prob_threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                class_id = int(obj[1])

                color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                det_label = labels_map[class_id] if labels_map else str(class_id)
                cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 1)

        return frame


if __name__ == '__main__':
    sys.exit(main() or 0)
