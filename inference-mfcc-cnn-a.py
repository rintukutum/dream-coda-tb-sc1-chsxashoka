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
	model1 = rt.InferenceSession(os.path.join(model_path,'model_fold_0.onnx'))
	onnx_pred1 = model1.run(None, {"input": input_df})
	model2 = rt.InferenceSession(os.path.join(model_path,'model_fold_1.onnx'))
	onnx_pred2 = model2.run(None, {"input": input_df})
	model3 = rt.InferenceSession(os.path.join(model_path,'model_fold_2.onnx'))
	onnx_pred3 = model3.run(None, {"input": input_df})
	model4 = rt.InferenceSession(os.path.join(model_path,'model_fold_3.onnx'))
	onnx_pred4 = model4.run(None, {"input": input_df})
	model5 = rt.InferenceSession(os.path.join(model_path,'model_fold_4.onnx'))
	onnx_pred5 = model5.run(None, {"input": input_df})
	return [onnx_pred1[0][0][1],onnx_pred2[0][0][1],onnx_pred3[0][0][1],onnx_pred4[0][0][1],onnx_pred5[0][0][1]]


def inference(path):
	test_df = pd.read_csv(path+'/test.csv')
	# Extract features from test data frame
	X_features = test_df.iloc[:, 2:82]
	Y_labels = test_df.iloc[:, -1]
	X_features = np.array(X_features.values.tolist())
	Y_labels = np.array(Y_labels.values.tolist())
	Y_labels = to_categorical(labelencoder.fit_transform(Y_labels))
	# Change datatype to double
	X_features = X_features.astype(np.float32)

	model_path = './models-b32-e100-lr-0_01/'
	
	predictions_0 = []
	predictions_1 = []
	predictions_2 = []
	predictions_3 = []
	predictions_4 = []
	for i in range(len(X_features)):
		t_0, t_1, t_2, t_3, t_4 = predict([X_features[i].reshape(80, 1)], model_path)
		predictions_0.append(t_0)
		predictions_1.append(t_1)
		predictions_2.append(t_2)
		predictions_3.append(t_3)
		predictions_4.append(t_4)

	y_pred = []
	y_pred.append(predictions_0)
	y_pred.append(predictions_1)
	y_pred.append(predictions_2)
	y_pred.append(predictions_3)
	y_pred.append(predictions_4)
	y_true = np.argmax(Y_labels, axis=1)

	for j in range(len(y_pred)):
		for i in range(len(y_pred[j])):
			if y_pred[j][i] > 0.5:
				y_pred[j][i] = 1
			else:
				y_pred[j][i] = 0

	for i in range(len(y_pred)):
		print('Fold: ', i)
		accuracy = np.sum(y_pred[i] == y_true) / len(y_true)
		print('Accuracy: ', accuracy)
		cm = confusion_matrix(y_true, y_pred[i])
		sns.heatmap(cm, annot=True, fmt='d')
		plt.show()


if __name__ == '__main__':
	path = './input/metadata/'
	inference(path)