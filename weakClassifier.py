# coding=utf-8
import numpy as np
from time import time

from setting import FACE, NON_FACE

class WeakClassifier(object):
    """
    @:param self.direction: when it is 1, the sample will be considered as face if its value less than threshold.
    """
    def __init__(self):
        self.direction   = None
        self.threshold   = None
        self.dimension   = None
        self.weightError = np.inf

    def fit(self, X, W, Y):
        """To minimize the weighted error function
        :param X: A matrix sampleNum * DimensionNum
        :param W: Weight corresponding to each sample DimensionNum*1
        :param Y: the label of each sample shape:sampleNum*1
        :return: minWeightError
        """
        dimensionNum = X.shape[1]
        FaceIndex    = np.where(Y==FACE)[0]
        NonFaceIndex = np.where(Y==NON_FACE)[0]
        FaceWeightSum    = W[FaceIndex].sum()
        NonFaceWeightSum = W[NonFaceIndex].sum()
        weakStartTime = time()
        predTem = np.zeros((Y.shape[0], 1))
        for dim in range(dimensionNum):
            if dim !=0  and dim %1000 == 0:
                print(str(dim) + "dim...")
            FaceWeightValSum    = (W[FaceIndex, 0] * X[FaceIndex, dim]).sum()
            NonFaceWeightValSum = (W[NonFaceIndex, 0] * X[NonFaceIndex, dim]).sum()

            threshold = (FaceWeightValSum/FaceWeightSum + NonFaceWeightValSum/NonFaceWeightSum) / 2
            for direction in [1, -1]:
                #
                # FaceWeightSumBeforeTh    = W[np.intersect1d(FaceIndex,
                #                         np.where(X[:, dim] <  threshold)[0])].sum()
                #
                # NonFaceWeightSumBeforeTh = W[np.intersect1d(NonFaceIndex,
                #                         np.where(X[:, dim] >= threshold)[0])].sum()
                #
                # tempWeightError = min(FaceWeightSumBeforeTh + (NonFaceWeightSum-NonFaceWeightSumBeforeTh),
                #                       NonFaceWeightSumBeforeTh + (FaceWeightSum-FaceWeightSumBeforeTh))

                predTem[X[:, dim] * direction < threshold * direction, :] = FACE
                predTem[X[:, dim] * direction >= threshold * direction, :] = NON_FACE
                tempWeightError = W[predTem != Y].sum()

                if tempWeightError < self.weightError:
                    self.weightError = tempWeightError
                    self.dimension = dim
                    self.direction = direction
                    self.threshold = threshold

        weakEndTime = time()
        print("weak.."+str(weakEndTime-weakStartTime))
        if self.weightError < 0.0001:
            self.weightError = 0.0001
        print(self.weightError)
        print("weakClassifier."+str(self.direction)+" "+str(self.threshold)+" "+str(self.dimension))
        return self.weightError

    def predict(self, X):
        pred = np.ones((X.shape[0], 1)) * NON_FACE
        pred[X[:, self.dimension] * self.direction < self.threshold * self.direction, :] = FACE

        return pred
