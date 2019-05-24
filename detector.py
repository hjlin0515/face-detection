# coding=utf-8
from PIL import Image
from matplotlib import image
from setting import WINDOW_WIDTH, WINDOW_HEIGHT, FACE, NON_FACE, TEST_RESULT_PIC, TEST_RESULT_INFO
from image import Img
import  numpy as np
from adaboost import Adaboost
from weakClassifier import WeakClassifier
from haar import Haar

from setting import WINDOW_HEIGHT, WINDOW_WIDTH

class Detector(object):

    def __init__(self, model):

        self.DETECT_START = 5.
        self.DETECT_END   = 9.
        self.DETECT_STEP  = 0.4
        self.DETECT_STEP_FACTOR = 10

        self.haar = Haar(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.model = model
        self.selectedFeatures = [None for i in range(model.n_estimators)]
        self._selectFeatures()

    def detectFace(self, fileName, _show=True, _save=False, _saveInfo=False):
        """
        :param fileName: 
        :param _show: 
        :param _save: 
        :return: 
        """
        img = Img(fileName, calIntegral=False)

        #scaledWindows: [[window_x, window_y, window_w, window_h, window_scale],...]
        scaledWindows = []

        for scale in np.arange(self.DETECT_START, self.DETECT_END, self.DETECT_STEP):
            self._detectInDiffScale(scale, img, scaledWindows)

        scaledWindows = np.array(scaledWindows)

        # detect whether the scaledWindows are face
        predWindow = self._detectScaledWindow(scaledWindows, img)

        #optimalWindow = self._optimalWindow(predWindow)

        mostProbWindow = self._getMostProbWindow(predWindow)

        if _show:
            self.show(img.mat, mostProbWindow)
        if _save:
            self.save(img.mat, mostProbWindow, fileName)
        if _saveInfo:
            self.saveProbWindowInfo(mostProbWindow, fileName)

    def show(self, imageMat, faceWindows):
        """show the result of detection
        :param imageMat:
        :param faceWindows:
        :return:
        """
        if faceWindows[0].shape[0] == 0:
            Image.fromarray(imageMat).show()
            return
        for i in range(len(faceWindows)):
            window_x, window_y, window_w, window_h, scale, prob = faceWindows[i]
            self._drawLine(imageMat, int(window_x), int(window_y), int(window_w), int(window_h))
        Image.fromarray(imageMat).show()

    def save(self, imageMat, faceWindows, originFileName):
        if faceWindows[0].shape[0] == 0:
            return
        for i in range(len(faceWindows)):
            window_x, window_y, window_w, window_h, scale, prob = faceWindows[i]
            self._drawLine(imageMat, int(window_x), int(window_y), int(window_w), int(window_h))
        Image.fromarray(imageMat).save((TEST_RESULT_PIC + "detected" +
                                        originFileName.split('/')[-1]).replace("pgm", "bmp") )

    def saveProbWindowInfo(self, window, originFileName):
        with open((TEST_RESULT_INFO  +
                                        originFileName.split('/')[-1]).replace("pgm", "pts"), "w") as f:
            if len(window[0] > 0 ):
                f.write(str(window[0][0]) + " " + str(window[0][1]) + " " + str(window[0][2]) + " " + str(window[0][3]))


    def _selectFeatures(self):
        """ select the features according to Adaboost classifier
        :return: [[haar_type, y, w, h, dimension],...]
        """
        for i in range(self.model.n_estimators):
            self.selectedFeatures[i] = self.haar.features[self.model.weakClassifiers[i].dimension] \
                                       + [self.model.weakClassifiers[i].dimension]

            # print(self.selectedFeatures[i])

    def _getMostProbWindow(self, predWindow):
        """ return the most likely one
        :param predWindow:
        :return:
        """
        mostProb = -np.inf
        mostProbWindow = np.array([])
        for i in predWindow:
            if i[-1] > mostProb:
                mostProbWindow = i
                mostProb = i[-1]
        print(mostProbWindow)
        return [mostProbWindow]

    def _drawLine(self, imageMat, x, y, w, h):
        """draw the boundary of the face in the image
        """
        imageMat[y,     x:x+w] = 0
        imageMat[y+h,   x:x+w] = 0
        imageMat[y:y+h, x    ] = 0
        imageMat[y:y+h, x+w  ] = 0

    def _detectInDiffScale(self, scale, img, scaledWindows):
        """
        :param scale:
        :param img:
        :param scaledWindows:
        :return:
        """
        SCALED_WINDOW_WIDTH  = int(WINDOW_WIDTH  * scale)
        SCALED_WINDOW_HEIGHT = int(WINDOW_HEIGHT * scale)

        scaled_window_x_limit = img.WIDTH  - SCALED_WINDOW_WIDTH
        scaled_window_y_limit = img.HEIGHT - SCALED_WINDOW_HEIGHT

        step = int(SCALED_WINDOW_WIDTH/self.DETECT_STEP_FACTOR)

        for x in range(0, scaled_window_x_limit, step):
            for y in range(0, scaled_window_y_limit, step):
                scaledWindows.append((x, y, SCALED_WINDOW_WIDTH, SCALED_WINDOW_HEIGHT, scale))

    def _detectScaledWindow(self, scaledWindows, img):
        """detect each of scaledWindow
        :param scaledWindows:
        :param img:
        :return:
        """
        scaledWindowsMat = np.zeros((scaledWindows.shape[0], len(self.haar.features)), dtype='float32')

        for window in range(scaledWindows.shape[0]):
            window_x, window_y, window_w, window_h, scale = scaledWindows[window]

            window_x, window_y, window_w, window_h = int(window_x), int(window_y), int(window_w), int(window_h)

            subWindowImg          = Img(mat=img.mat[window_y : window_y+window_h, \
                                           window_x : window_x+window_w])
            subWindowImgIntegral  = subWindowImg.integralMat

            # #normalization
            # sumVal        = sum(sum(subWindowImg.mat[y:y+h, x:x+w]))
            # sqSumVal      = sum(sum(subWindowImg.mat[y:y+h, x:x+w] ** 2))
            # meanVal       = sumVal   / (w * h)
            # sqMeanVal     = sqSumVal / (w * h)
            # normFactorVal = np.sqrt(sqMeanVal - meanVal ** 2)
            #
            # if normFactorVal == 0:
            #     normFactorVal = 1

            for f in range(len(self.selectedFeatures)):
                h_type, x, y, w, h, dimension = self.selectedFeatures[f]
                x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

                if h_type == "HAAR_TYPE_I":
                    pos = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    neg = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                    scaledWindowsMat[window][dimension] = (pos - neg) / (2 * w * h)
                elif h_type == "HAAR_TYPE_II":
                    neg = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg) / (2 * w * h)
                elif h_type == "HAAR_TYPE_III":
                    neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos  = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)
                    neg2 = self.haar.getPixelValInIntegralMat(x + 2 * w, y, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg1 - neg2) / (3 * w * h)

                elif h_type == "HAAR_TYPE_IV":
                    neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos  = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                    neg2 = self.haar.getPixelValInIntegralMat(x, y + 2 * h, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg1 - neg2) / (3 * w * h)

                elif h_type == "HAAR_TYPE_V":
                    neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos1 = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)
                    pos2 = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                    neg2 = self.haar.getPixelValInIntegralMat(x + w, y + h, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos1 + pos2 - neg1 - neg2) / (4 * w * h)

        pred = self.model.predict_prob(scaledWindowsMat)
        indexs = np.where(pred > 0)[0]
        predWindow = np.zeros((len(indexs), scaledWindows.shape[1]+1), dtype=object)
        for i in range(len(indexs)):
            predWindow[i] = np.append(scaledWindows[indexs[i]], pred[indexs[i]])

        return predWindow

    def _optimalWindow(self, predWindow):
        """optimize the windows according to the situations of overlapping...
        :param predWindow: (x, y, w, h, scale, prob)
        :return:
        """
        optimalWindowMap = np.array([i for i in range(predWindow.shape[0])])

        for i in range(predWindow.shape[0]):
            for j in range(i+1, predWindow.shape[0]):
                overlap = False
                contain = False

                if self._windowInAnotherWindow(predWindow[i], predWindow[j]):
                    # optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[i])] = optimalWindowMap[j]
                    contain = True
                elif self._windowInAnotherWindow(predWindow[j], predWindow[i]):
                    # optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[j])] = optimalWindowMap[i]
                    contain = True
                else:
                    for x in [predWindow[i][0], predWindow[i][0] + predWindow[i][2]]:
                        for y in [predWindow[i][1], predWindow[i][1] + predWindow[i][3]]:
                            if self._pointInWindow((x, y), predWindow[j]):
                                overlap = True
                                break
                    for x in [predWindow[j][0], predWindow[j][0] + predWindow[j][2]]:
                        for y in [predWindow[j][1], predWindow[j][1] + predWindow[j][3]]:
                            if self._pointInWindow((x, y), predWindow[i]):
                                overlap = True
                                break

                if overlap or contain:
                    if predWindow[i][-1] == max(predWindow[i][-1], predWindow[j][-1]):
                        optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[j])] = optimalWindowMap[i]

                    else:
                        optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[i])] = optimalWindowMap[j]

        optimalWindow = np.zeros(len(set(optimalWindowMap)), dtype=object)
        index = 0
        for i in set(optimalWindowMap):
            optimalWindow[index] = predWindow[i]
            index = index + 1
        return optimalWindow

    def _pointInWindow(self, point, window):
        """
        :param point: (x, y)
        :param window: (x, y, w, h, scale, prob)
        :return:
        """
        if point[0] >= window[0] and point[0] <= window[0] + window[2]:
            if point[1] >= window[1] and point[1] <= window[1] + window[3]:
                return True
        return False

    def _windowInAnotherWindow(self, window, anotherWindow):
        """
        :param window: (x, y, w, h, scale, prob)
        :param anotherWindow: (x, y, w, h, scale, prob)
        :return:
        """
        if self._pointInWindow((window[0], window[1]), anotherWindow):
            if self._pointInWindow((window[0]+window[2], window[1]), anotherWindow):
                if self._pointInWindow((window[0], window[1]+window[3]), anotherWindow):
                    if self._pointInWindow((window[0]+window[2], window[1]+window[3]), anotherWindow):
                        return True
        return False



