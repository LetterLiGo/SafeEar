#!/bin/bash  
  
rank=4
# Set the base command  
base_command="python examples/hubert/simple_kmeans/dump_mfcc_feature.py examples/hubert/ train ${rank}"  
  
# Set the output directory  
output_dir="examples/hubert/mfcc_feature"  
  
# Loop from 0 to 3  
for i in {0..3}; do  
    # Construct the full command with the current value of i  
    full_command="${base_command} ${i} ${output_dir}"  
      
    # Run the command  
    echo "Running command: ${full_command}"  
    ${full_command}  
done  
  
echo "Done."  
