from __future__ import print_function

import numpy as np
import cv2 as cv


class LetterStatModel(object):
    class_n = 26
    train_ratio = 0.5

    def load(self, fn):
        self.model = self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1 # index is based on 26 letters in this case
        return new_responses

class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv.ml.ANN_MLP_create()

    def train(self, samples, responses):
        _sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)
        layer_sizes = np.int32([var_n, 100, 100, self.class_n])

        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0.0)
        self.model.setBackpropWeightScale(0.001)
        self.model.setTermCriteria((cv.TERM_CRITERIA_COUNT, 20, 0.01))
        self.model.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
        self.model.train(samples, cv.ml.ROW_SAMPLE, np.float32(new_responses))

    def predict(self, samples):
        _ret, resp = self.model.predict(samples)
        return resp.argmax(-1)



class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv.ml.Boost_create()

    def train(self, samples, responses):
        _sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv.ml.VAR_NUMERICAL] * var_n + [cv.ml.VAR_CATEGORICAL, cv.ml.VAR_CATEGORICAL], np.uint8)

        self.model.setWeakCount(15)
        self.model.setMaxDepth(10)
        self.model.train(cv.ml.TrainData_create(new_samples, cv.ml.ROW_SAMPLE, new_responses.astype(int), varType = var_types))

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        _ret, resp = self.model.predict(new_samples)

        return resp.ravel().reshape(-1, self.class_n).argmax(1)
