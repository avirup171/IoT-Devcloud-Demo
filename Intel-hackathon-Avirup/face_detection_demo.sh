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

cd $PBS_O_WORKDIR

python3 main.py -m models/frozen_inference_graph.xml \
                        -d $DEVICE \
                        -i $INPUT_FILE \
                        -o $OUTPUT_FILE \
                        -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so
