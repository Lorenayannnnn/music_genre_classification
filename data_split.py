import os
import random

rootdir = '../Data/images_original'
folders = [x[0] for x in os.walk(rootdir)]
folders.pop(0)

file1 = open('data_split.txt', 'w+')

ttrain = []
tdev = []
ttest = []

for folder in folders:
    dir_list = os.listdir(folder)
    name = folder[24:]
    new = []
    for item in dir_list:
        new.append(name + '/' + item)
    random.shuffle(new)
    train = new[0:69]
    dev = new[70:89]
    test = new[90:99]
    ttrain += train
    tdev += dev
    ttest += test

file1.write('train:\n')
file1.write(str(ttrain) + '\n')
file1.write('dev:\n')
file1.write(str(tdev) + '\n')
file1.write('test:\n')
file1.write(str(ttest) + '\n')

file1.close()
