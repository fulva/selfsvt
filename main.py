import os
import numpy as np
import torch as th
import random
import argparse

from models import ObservationEncoder
from train import train_siamese

random.seed(10)


parser = argparse.ArgumentParser(description='SelfSVT Training')

parser.add_argument('--w-size', default=32, type=int, help='The size of the input image.')
parser.add_argument('--num', default=8, type=int, help='The number of switches.')
parser.add_argument('--seen', default=25, type=int, help='The proportion of different states included in training.')
parser.add_argument('--scene', default='EnR', type=str, help='The name of the environment.')


def main():
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("----------"+str(device) + " will be used to do the computation"+"----------")

    dirpath = os.path.join(os.path.join(os.getcwd(), "data/"), args.scene)
    data_dict = np.load(os.path.join(dirpath, args.scene+"_all_images_N"+str(args.num)+"_Size"+str(args.w_size)+".npy"), allow_pickle=True).item()
    states =  list(data_dict.keys())
    random.shuffle(states)        
    print("    The name of our environment is: "+ str(args.scene))
    print("    The number of switches: "+str(args.num))
    print("    The size of our input image: " + str(list(data_dict.values())[0].shape))
    print("    The number of total different states: "+ str(len(states)))
    print("    " + str(args.seen) + "% states will be included in training process.")
    print("    " + str(100-args.seen) + "% states will be included in test process.")

    F = ObservationEncoder(args.num, args.w_size)
    if th.cuda.is_available():
        F.cuda()
    train_siamese(F, states, data_dict, device, args.scene, args.num, args.seen, steps = 10000, bs = 512)
    
    if not os.path.exists(os.path.join(os.getcwd(), "output/")):
        os.makedirs(os.path.join(os.getcwd(), "output/"))


    th.save(F, 'output/ImageToState' +"_"+args.scene+"_N"+str(args.num)+"_Size"+str(args.w_size)+ "_Seen"+str(args.seen))
    del F
    th.cuda.empty_cache()

if __name__ == '__main__':
    main()
