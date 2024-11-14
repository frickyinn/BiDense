import os
import subprocess


methods = ['adabin', 'bisrnet']
for m in methods:
    subprocess.run(['python', 'train_segmentation.py', f'configs/segmentation/upernet_{m}_pascal_voc.yaml'])
# methods = ['fp32', 'fp32_0', 'bnn', 'react', 'adabin', 'bisrnet', 'bidense']
# for m in methods:
#     subprocess.run(['python', 'train_depth.py', f'configs/depth/upernet_{m}_kitti.yaml'])

# methods = ['fp32', 'fp32_0', 'bnn', 'react', 'adabin', 'bisrnet', 'bidense']
# for m in methods:
#     subprocess.run(['python', 'train_depth.py', f'configs/depth/upernet_{m}_nyu.yaml'])