#!/bin/bash


# Assign input arguments to variables
input_dim="(32,32)"
weight_dim="(32,32)"
output_dim="(32,32)"

# Run the Python script with the provided dimensions
cd /app/Deeploy/DeeployTest/Tests/testFloatGEMM
python3 testFloatGEMMgenerate.py "$input_dim" "$weight_dim" "$output_dim"

cd /app/Deeploy/DeeployTest/TEST_SIRACUSA/build
make clean


cd /app/Deeploy/DeeployTest
# python testRunner_tiled_siracusa.py -t Tests/testFloatGEMM --cores=1 --l1=6000 --defaultMemLevel=L2 --profileTiling L2 --doublebuffer
# python testRunner_siracusa.py -t Tests/testFloatGEMM --cores=1 
# python testRunner_tiled_siracusa.py -t Tests/testFloatGEMM --cores=8 --l1=10000 --defaultMemLevel=L2 --profileTiling L2

python testRunner_tiled_siracusa.py -t Tests/testFloatGEMM --cores=8 --l1 8000 --defaultMemLevel=L2 --doublebuffer