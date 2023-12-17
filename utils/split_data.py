from pathlib import Path
from sklearn.model_selection import  StratifiedShuffleSplit
import shutil
import numpy as np
import math
import os
import random
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

def image_train_test_split(path, fmt, train_size):
    train_folder = Path('train')
    test_folder = Path('test')

    train_folder.mkdir(exist_ok=True)
    test_folder.mkdir(exist_ok=True)

    data_path = Path(path)
    data = []
    for d in data_path.glob('*'):
        print(d)
        for f in d.glob(f'*.{fmt}'):
            print(f)
            data.append([f, d.stem])
            print(data)
    #   data = pd.DataFrame(data)
    print(data)
    ss = StratifiedShuffleSplit(1, train_size=0.8)
    train_ix, test_ix = next(ss.split(data[:,0],data[:,1]))

    train_set, test_set = data[train_ix], data[test_ix]

    for p, c in train_set:

        (train_folder / c).mkdir(exist_ok=True)
        shutil.move(p, train_folder.joinpath(*p.parts[-2:]))

    for p, c in test_set:

        (test_folder / c).mkdir(exist_ok=True)
        shutil.move(p, test_folder.joinpath(*p.parts[-2:]))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root path for the Rareplanes dataset")
    parser.add_argument("--train-percent", type=float, default=0.8)
    args = parser.parse_args()
    p = args.data_root+'_split'
    isExist = os.path.exists(args.data_root+'_split')
    if not isExist:
   # Create a new directory because it does not exist
        os.makedirs(p)
        print("The new directory is created!")
    data = os.listdir(args.data_root)
    print(len(data))
    train, valid = train_test_split(data, test_size=0.2, random_state=1)
    print(len(train))

    # image_train_test_split(args.data_root,args.data_root+'_split',args.train_percent)


if __name__ == '__main__':
    main()