import numpy as np
import glob
import cv2
import pdb

class Perceptron(object):
    # x: reduced emoji images: 20 x 400 (original size 200 x 200, resize it to be 20 x 20)
    # y: class of emoji:       20 x 1   (+1 for smiling and -1 for non-smiling)
    # alpha: weights for each pixel: 400 x 1
    # imageWeights: how much of alpha is composed from each image: 20 x 1
    def __init__(self, x, y):
        self.x = x.reshape((x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        self.y = y
        self.alpha = np.zeros((self.x.shape[1], 1))
        self.imageWeights = np.zeros((self.y.shape[0], 1))

    '''
    Forward function returns 2 variables:
    predictions: raw value of perceptron output
    accuracy: the 0-1 accuracy of our predictions.
    '''
    def forward(self):
        predictions = np.vectorize(Perceptron.binarize)(np.matmul(self.x, self.alpha))
        accuracy = np.sum(np.equal(predictions, self.y)) / self.y.shape[0]
        return predictions, accuracy

    '''
    Backward function updates alpha weights and has no return
    '''
    def backward(self, predictions):
        updateDirection = np.multiply(self.y, np.not_equal(predictions, self.y))  # from {-1, 0, 1} to update
        self.imageWeights += updateDirection;
        updateVector = np.matmul(np.transpose(self.x), updateDirection)
        self.alpha = self.alpha + updateVector

    '''
    Helper function that binarizes an input into {-1, 1} based on sign
    '''
    @staticmethod
    def binarize(f):
        return 1 if f > 0 else -1


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
    # pdb.set_trace()
    max_iters = 100
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
        # Finish if accuracy maxes out
        if accuracy == 1.:
            print('Converged after {} epochs'.format(i))
            break

    # Output weights of each image
    print('Weight of each image: \n{}'.format(p.imageWeights))


