#PBS
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4
NUM_INFER_REQ=$5
NUM_ITERATIONS=$6

if [ "$2" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/altera/aocl-pro-rte/aclrte-linux64/
    source /opt/fpga_support_files/setup_env.sh
    aocl program acl0 /opt/intel/computer_vision_sdk/bitstreams/a10_vision_design_bitstreams/5-0_PL1_FP11_ResNet.aocx
    
fi
SAMPLEPATH=$PBS_O_WORKDIR

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/build/intel64/Release/lib/ 
MODEL_ROOT=/opt/intel/computer_vision_sdk/deployment_tools/intel_models

python3 main.py  -d $DEVICE \
                 -i $INPUT_FILE \
                 -l /opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so \
                 -o $OUTPUT_FILE \
                 -m ${SAMPLEPATH}/mobilenet-ssd/${FP_MODEL}/mobilenet-ssd.xml \
                 -nireq $NUM_INFER_REQ \
                 -niter $NUM_ITERATIONS
