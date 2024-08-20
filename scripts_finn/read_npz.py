# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This file is subject to the Xilinx Design License Agreement located
# in the LICENSE.md file in the root directory of this repository.
#
# This file contains confidential and proprietary information of Xilinx, Inc.
# and is protected under U.S. and international copyright and other
# intellectual property laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any rights to the materials
# distributed herewith. Except as otherwise provided in a valid license issued to
# you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
# MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
# DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
# INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
# FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
# in contract or tort, including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature related to, arising
# under or in connection with these materials, including for any direct, or any
# indirect, special, incidental, or consequential loss or damage (including loss
# of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the possibility of the
# same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-safe, or for use in
# any application requiring failsafe performance, such as life-support or safety
# devices or systems, Class III medical devices, nuclear facilities, applications
# related to the deployment of airbags, or any other applications that could lead
# to death, personal injury, or severe property or environmental damage
# (individually and collectively, "Critical Applications"). Customer assumes the
# sole risk and liability of any use of Xilinx products in Critical Applications,
# subject only to applicable laws and regulations governing limitations on product
# liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.
import os
import sys

in_path = 'global_expected_full.npz'
out_path = 'int_results_'
array_name = sys.argv[1]

import numpy as np

data = np.load(in_path)

# Print all array names in the .npz file
print("Array names in the .npz file:")
for key in data.files:
    print(key)

# Function to convert an int8 value to signed hexadecimal
def int8_to_signed_hex(val):
    # Convert int8 to signed hexadecimal representation
    hex_val = format(np.uint8(val), '02x') 
    return hex_val

def int32_to_signed_hex(val):
    # Convert int32 to signed hexadecimal representation
    hex_val = format(np.uint32(val), '08x')
    return hex_val

# Check if the specified array is in the file
if array_name not in data.files:
    print(f"Array '{array_name}' not found in the file.")
else:
    # Get the specified array
    array = data[array_name]
    
    # Open a text file to write the output
    with open(out_path + array_name + '.txt', 'w') as f:

        # Write each int8 value as signed hexadecimal to the file
        for val in np.nditer(array):

            #hex_val = int32_to_signed_hex(val)
            hex_val = int8_to_signed_hex(val)
            f.write(hex_val + '\n')
    
    print(f"Data from array '{array_name}' written")
