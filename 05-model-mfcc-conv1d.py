import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from datetime import datetime
import tf2onnx

def keras_2_onnx(model, filename):
    out_path = filename + ".onnx"
    spec = (tf.TensorSpec((None, 80, 1), tf.float32, name="input"),)
    m_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=out_path)
    out_name = [n.name for n in m_proto.graph.output]
    return out_name


def model_conv1d():
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    # dump X_train, X_test csv file
    for i in range(5):
        temp_df_0 = pd.DataFrame()
        temp_df_1 = pd.DataFrame()
        temp_df_0 = pd.read_csv('./input/metadata/X_train_Fold_{}.csv'.format(i))#, index=False)
        temp_df_1 = pd.read_csv('./input/metadata/X_test_Fold_{}.csv'.format(i))#, index=False)
        X_train.append(temp_df_0.iloc[:, 2:82])
        X_test.append(temp_df_1.iloc[:, 2:82])
        y_train.append(temp_df_0.iloc[:,-1])
        y_test.append(temp_df_1.iloc[:,-1])

    # # Comvert data frames to numpy array
    for i in range(5):
        X_train[i] = np.array(X_train[i].values.tolist())  
        y_train[i] = np.array(y_train[i].values.tolist())
        X_test[i] = np.array(X_test[i].values.tolist())
        y_test[i] = np.array(y_test[i].values.tolist())

    labelencoder = LabelEncoder()
    for i in range(5):#(kf.get_n_splits(train)):
        y_train[i] = to_categorical(labelencoder.fit_transform(y_train[i]))
        y_test[i] = to_categorical(labelencoder.fit_transform(y_test[i]))

    ####################### Create the model #######################
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
    model = Sequential()
    model.add(tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(80, 1)))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Conv1D(64, 2, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Conv1D(128, 2, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    
    ####################### model fit #######################
    num_epochs = 100
    num_batch_size = 32
    history = []
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./callbacks-cnn-b{num_batch_size}-e{num_epochs}-lr-0_01/', histogram_freq=1)
    start = datetime.now()
    for i in range(5):#(kf.get_n_splits(train)):
        print("Fold: ", i)
        history.append(model.fit(X_train[i], y_train[i], epochs=num_epochs, batch_size=num_batch_size, validation_data=(X_test[i], y_test[i]), callbacks=[tensorboard_callback]))
        
        outpath_mod = f'./models-cnn-b{num_batch_size}-e{num_epochs}-lr-0_01/'
        filename = os.path.join(outpath_mod,f'model_fold_{str(i)}')
        print("Saving Model ", i)
        keras_2_onnx(model, filename)
        
    duration = datetime.now() - start
    print("Training Completed in: ", duration)

    ####################### model stat #######################
    accuracy = []
    loss = []
    val_accuracy = []
    val_loss = []
    for i in range(5):#(kf.get_n_splits(train)):
        accuracy.append(np.mean(history[i].history['accuracy']))
        loss.append(np.mean(history[i].history['loss']))
        val_accuracy.append(np.mean(history[i].history['val_accuracy']))
        val_loss.append(np.mean(history[i].history['val_loss']))

    # Printing the mean values
    print("Mean accuracy: ", np.mean(accuracy))
    print("Mean loss: ", np.mean(loss))
    print("Mean val_accuracy: ", np.mean(val_accuracy))
    print("Mean val_loss: ", np.mean(val_loss))

if __name__ == "__main__":
    model_conv1d()