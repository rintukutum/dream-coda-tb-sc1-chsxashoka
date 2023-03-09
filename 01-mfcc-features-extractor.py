'''
Inputs:
1. file_name: path to file 
OUTPUT
1. mfcc_all: numpy array of 40x22
2. mfcc_mean_features: numpy array of 40x2, having mean features and mean lables
3. mfcc_std_features: numpy array of 40x2, having standard deviation features and standard deviation lables
4. mfcc_features: pandas data frame of 1x80, having mean and standard deviation features
'''

import librosa
import numpy as np
import pandas as pd
import os

def features_extractor(file_name):
    audio_data, sample_rate = librosa.load(file_name)#, res_type='kaiser_fast')  

    mfcc_all = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    
    # Mean Features
    mean_features = np.mean(mfcc_all.T, axis=0)
    # add lables as m1, m2, ..
    mean_class_lables = ['m'+str(i) for i in range(1,41)]
    # np array of mean features and lables
    mfcc_mean_features = np.array([mean_features, mean_class_lables])
    # pd Data frame
    mfcc_mean_scaled = pd.DataFrame(mean_features, index=mean_class_lables).T

    standard_featues = np.std(mfcc_all.T, axis=0)
    # add lables as s1, s2, ..
    standard_class_lables = ['s'+str(i) for i in range(1,41)]
    # np array of standard features and lables
    mfcc_std_features = np.array([standard_featues, standard_class_lables])
    # pd data frame
    mfcc_std_scaled = pd.DataFrame(standard_featues, index=standard_class_lables).T

    # append mean and standard deviation features
    mfcc_features = pd.concat([mfcc_mean_scaled, mfcc_std_scaled], axis=1)

    return mfcc_all, mfcc_mean_features, mfcc_std_features, mfcc_features

################################

if __name__ == '__main__':
    audio_dataset_path = './input/raw_data/'
    audio_metadata = './input/metadata/CODA_TB_Solicited_Meta_Info.csv'
    clinical_metadata = './input/metadata/CODA_TB_Clinical_Meta_Info.csv'

    metadata = pd.read_csv(audio_metadata)
    metadata = metadata.drop(metadata.columns[-1], axis=1)

    ## Create a data frame with all extracted features
    extracted_features = pd.DataFrame()
    for index_num, row in metadata.iterrows():
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'{}'.format(row["filename"]))
        mfccs, mfcc_mean_features, mfcc_std_features, mfcc_features = features_extractor(file_name)
        extracted_features = pd.concat([extracted_features, mfcc_features], axis=0)
    # Index the data frame
    extracted_features.index = range(0, len(extracted_features))
    # join two data frame extracted_features and metadata in column wise, reset index
    final_extracted_features = pd.concat([metadata, extracted_features], axis=1)
    # clinical data
    Clinial_Info = pd.read_csv(clinical_metadata)
    # extract first and last columns
    tb_status = Clinial_Info.iloc[:,[0, -1]]
    # add "tb_status" column to final_extracted_features data frame as per "participant" column
    final_extracted_features = pd.merge(final_extracted_features, tb_status, on='participant')
    ## Save the data frame to csv file at metadata folder
    final_extracted_features.to_csv('./input/metadata/final_extracted_features.csv', index=False)