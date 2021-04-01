import regex as re
import pandas as pd
import numpy as np
from os import listdir


#this function takes a file name such as 01_John_0998 based on regex get 'John' from it
def get_label(filename):

    found = re.search('_(.+?)_', filename)

    if found:
        file_name = found.group(1)

    return file_name


#This function is used to get a directory and collect all files from that dir and get_labels() from them based on a regex,
#another parameter is file name it takes a file name and save with this file name

def get_Classes(dir , Name_of_file):
    label_array = np.empty(0)
    id_array = np.empty(0)

    file_names = [f for f in listdir(dir)]

    for file_name in file_names:
        absolute_path = dir + '/' + file_name
        labels = get_label(file_name)
        label_array = np.append(label_array, [labels])

# wtr = csv.writer(open ('out.csv', 'w'), delimiter=',', lineterminator='\n')

    for x in range(len(label_array)):
        id_array = np.append(id_array, [x])

    df = pd.DataFrame(list(zip(*[id_array, label_array])), columns=['Id', 'Name'])

    df.to_csv(Name_of_file+'.csv', index=False)

# for y in label_array :
#     wtr.writerow ([id_array] ,[y])
    return None


def extract_mfcc(audio):

    x,sr = librosa.load(audio,res_type='kaiser_fast')
    # librosa.load(audio, rate)
    mfccs = np.mean(librosa.feature.mfcc(x, sr,n_mfcc=40).T, axis=0)
    # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    return np.asarray(mfccs, dtype = 'float32')



def get_feature_file(dir, filename):


    wtr = csv.writer(open(filename+'.csv', 'a'), delimiter=',', lineterminator='\n')
    file_names = [f for f in listdir(dir)]

    li = []
    for i in range(40):
        li.insert(i, ("mfcc" + str(i + 1)))

    wtr.writerow(li)


    for file_name in file_names:
        absolute_path = dir + '/' + file_name
        feature = extract_mfcc(absolute_path)
        wtr.writerow(feature)
    return None

