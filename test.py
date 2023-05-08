import os
import torch
from pathlib import Path
from utils.data_loading import SonarDataset
import math
import numpy as np

dir_heats = Path('../plet/labels/')
dir_yaw = Path('../plet/yaw/')


def checkClassDistribution(angle_txt, discretized_size) :

    with open(angle_txt) as f:
        yaw = float(f.readlines()[0].replace('\n', '')) 
    
    step_cell = ( math.pi * 2 )/ float(discretized_size)
    yaw = yaw if yaw >= 0 else math.pi * 2 + yaw

    arr_idx = int( yaw / step_cell ) % discretized_size

    return arr_idx


def main() :

    #tensor = SonarDataset.preprocessYaw(dir_yaw, 180)
    #print(tensor)

    yaws =  os.listdir(dir_yaw)
    # heats = os.listdir(dir_heats)

    discretized_size = 36
    bucket = np.zeros(discretized_size)

    for name in yaws :
        idx = checkClassDistribution(str(dir_yaw) + '/' + name, discretized_size)
        bucket[idx] += 1
    
    print(f'Classes distributions non-normalized : \n{bucket}')

    bucket = bucket / bucket.sum()

    print(f'Classes distributions : \n{bucket}\nUniform distribution = {1/36.0}')
    '''
    print(f'Tot heats  = {len(heats)}')
    print(f'Tot yaws  = {len(yaws)}')

    FAKE YAW DATASET
    for idx in range(len(heats)) :
        filename = str(dir_yaw) + '/' + str(idx) + '.txt'
        with open(filename, 'w') as f:
            f.write('0.0')
    '''


if __name__ == '__main__' :
    main()