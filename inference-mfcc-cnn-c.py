import onnxruntime as rt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
labelencoder = LabelEncoder()

def predict(input_df, model_path):
	model5 = rt.InferenceSession(os.path.join(model_path,'model_fold_4.onnx'))
	onnx_pred5 = model5.run(None, {"input": input_df})
	return onnx_pred5[0][0][1]


def inference(path):
    test_df = pd.read_csv(path+'/test.csv')
    # Extract features from test data frame
    X_features = test_df.iloc[:, 2:82]
    Y_labels = test_df.iloc[:, -1]
    X_features = np.array(X_features.values.tolist())
    Y_labels = np.array(Y_labels.values.tolist())
    Y_labels = to_categorical(labelencoder.fit_transform(Y_labels))
    X_features = X_features.astype(np.float32)
    model_path = './models-b32-e100-lr-0_01/'
    predictions= []
    for i in range(len(X_features)):
        predictions.append(predict([X_features[i].reshape(80, 1)], model_path))

    y_true = np.argmax(Y_labels, axis=1)
    for i in range(len(predictions)):
        if predictions[i] > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    accuracy = np.sum(predictions == y_true) / len(y_true)
    print('Accuracy: ', accuracy)
    cm = confusion_matrix(y_true, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

if __name__ == '__main__':
	path = './input/metadata/'
	inference(path)