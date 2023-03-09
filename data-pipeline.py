import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def datapipeline(path):
    final_extracted_features = pd.read_csv(path)
    # Randomly shuffle the data frame and freeze the index
    final_extracted_features = final_extracted_features.sample(frac=1)#.reset_index(drop=False)

    # Split the data into train and test - 80% train and 20% test
    train = final_extracted_features[:int(0.8*(len(final_extracted_features)))]
    test = final_extracted_features[int(0.8*(len(final_extracted_features))):]

    # Dump the train and test data into csv files
    # dump test to csv file
    test.to_csv('./input/metadata/test.csv', index=False)

    # Generate 5 fold cross validation through sklearn on train data
    kf = KFold(n_splits=5, shuffle=True)

    # Create list of lists to store train and test data
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for train_index, test_index in kf.split(train):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train.append(train.iloc[train_index])
        X_test.append(train.iloc[test_index])
        y_train.append(train.iloc[train_index])
        y_test.append(train.iloc[test_index])

    # Dump X_train, X_test, y_train, y_test to csv files
    for i in range(kf.get_n_splits(train)):
        X_test[i].to_csv('./input/metadata/X_test_Fold_{}.csv'.format(i), index=False)
        X_train[i].to_csv('./input/metadata/X_train_Fold_{}.csv'.format(i), index=False)

if __name__ == '__main__':
    feature_path = './input/metadata/final_extracted_features.csv'
    datapipeline(feature_path)