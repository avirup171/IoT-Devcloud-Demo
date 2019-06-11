#PBS
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4

if [ "$2" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/altera/aocl-pro-rte/aclrte-linux64/
    source /opt/fpga_support_files/setup_env.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/5-0_PL1_FP11_ResNet.aocx
    
fi

if [ "$FP_MODEL" = "FP16" ]; then
  FPEXT='-fp16'
fi

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/openvino/deployment_tools/inference_engine/samples/build/intel64/Release/lib/ 
MODEL_ROOT=/opt/intel/openvino/deployment_tools/intel_models

python3 main.py  -m_fc models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001${FPEXT}.xml \
                        -m_ag models/Retail/object_attributes/age_gender/dldt/age-gender-recognition-retail-0013${FPEXT}.xml \
                        -m_em models/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003${FPEXT}.xml \
                        -m_hp models/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001${FPEXT}.xml \
                        -m_lm models/Transportation/object_attributes/facial_landmarks/custom-35-facial-landmarks/dldt/facial-landmarks-35-adas-0002$(FPTEXT).xml \
                        -d $DEVICE \
                       -i $INPUT_FILE \
                        -o $OUTPUT_FILE \
                        -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so
