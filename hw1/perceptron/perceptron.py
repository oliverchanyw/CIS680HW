import numpy as np
import glob
import cv2
import pdb

class Perceptron(object):
	# x: reduced emoji images: 20 x 400 (original size 200 x 200, resize it to be 20 x 20)
	# y: class of emoji:       20 x 1   (+1 for smiling and -1 for non-smiling)
	'''
	At initialization, create you alpha weights, something like "self.alpha = np.zeros([SIZE])"
	'''
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.alpha = None

	'''
	For forward function, you have the following steps
	1. make predictions for  self.x 
	2. compute accuracy with self.y
	'''
	def forward():
		predictions = None # np.ndarray of size (20, 1)
		accuracy = None    # float/double 
		return predictions, accuracy
	'''
	For backward function, you will update the parameters (alpha in this case) and you do not have to return anything
	'''
	def backward(predictions):
		return

# Please do not change how to data is organized as data loading depends on it
def load_emoji(data_dir='data', size=(20, 20)):
	file_names = glob.glob('{}/*/*.*'.format(data_dir))
	img_arr, lab_arr, reduced = [], [], []
	for file_name in file_names:
		_, face_type, name = file_name.split('/')
		img, lab = cv2.imread(file_name), -1
		reduced.append(cv2.resize(img, size, interpolation = cv2.INTER_CUBIC))
		if face_type == 'train_smile': 
			lab = 1
		img_arr.append(img)
		lab_arr.append(lab)
	return np.stack(reduced), np.stack(img_arr), np.stack(lab_arr)[...,None]

# the following is a dummy code showing how can load data and how you MAY train, NOT meant to run
if __name__ == '__main__':
	# LOAD DATA HERE...
	x, full_res, y = load_emoji() 
	pdb.set_trace()
	max_iters = 1000
	# CREATE PERCEPTRON
	p = Perceptron(x, y)
	# OVER EACH ITERATION
	for i in range(max_iters):
		# Predict
		predictions, accuracy = p.forward()
		# Update
		p.backward(predictions)
		# Report 
		print('Accuracy: {}'.format(accuracy))






























