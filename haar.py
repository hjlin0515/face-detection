# coding=utf-8
import numpy as np

"""
Haar Feature TYPE

     --- ---
    |   +   |
    |-------|
    |   -   |
     -------
        I
     --- ---
    |   |   |
    | - | + |
    |   |   |
     --- ---
       II

     -- -- --
    |  |  |  |
    |- | +| -|
    |  |  |  |
     -- -- --
       III

     --- ---
    |___-___|
    |___+___|
    |___-___|
       IV

     --- ---
    | - | + |
    |___|___|
    | + | - |
    |___|___|
        V
"""

class Haar(object):
    def __init__(self, img_width, img_height):
        self.IMG_WIDTH  = img_width
        self.IMG_HEIGHT = img_height

        self.WINDOW_WIDTH  = self.IMG_WIDTH
        self.WINDOW_HEIGHT = self.IMG_HEIGHT

        self.HAAR_TYPES = (
            'HAAR_TYPE_I',
            'HAAR_TYPE_II',
            'HAAR_TYPE_III',
            'HAAR_TYPE_IV',
            'HAAR_TYPE_V',
        )

        self.features = []
        self._createFeatures()

    def _createFeatures(self):
        """create all kinds of haar features in this window size
        :return: [(h_type, x, y, w, h),
                  (h_type, x, y, w, h),
                  ...]
                  notice: x,y are the coordinates in the image Matrix instead of the integral image Matrix
        """

        WIDTH_LIMIT  = {
            "HAAR_TYPE_I"   : self.WINDOW_WIDTH,
            "HAAR_TYPE_II"  : int(self.WINDOW_WIDTH/2),
            "HAAR_TYPE_III" : int(self.WINDOW_WIDTH/3),
            "HAAR_TYPE_IV"  : self.WINDOW_WIDTH,
            "HAAR_TYPE_V"   : int(self.WINDOW_WIDTH/2)
        }

        HEIGHT_LIMIT = {
            "HAAR_TYPE_I"   : int(self.WINDOW_HEIGHT/2),
            "HAAR_TYPE_II"  : self.WINDOW_HEIGHT,
            "HAAR_TYPE_III" : self.WINDOW_HEIGHT,
            "HAAR_TYPE_IV"  : int(self.WINDOW_HEIGHT/3),
            "HAAR_TYPE_V"   : int(self.WINDOW_HEIGHT/2)
        }

        for h_type in self.HAAR_TYPES:
            for w in range(1, WIDTH_LIMIT[h_type]+1):
                for h in range(1, HEIGHT_LIMIT[h_type]+1):

                    if h_type == "HAAR_TYPE_I":
                        x_limit = self.WINDOW_WIDTH  - w
                        y_limit = self.WINDOW_HEIGHT - 2*h

                        for x in range(0, x_limit+1):
                            for y in range(0, y_limit+1):
                                self.features.append([h_type, x, y, w, h])

                    if h_type == "HAAR_TYPE_II":
                        x_limit = self.WINDOW_WIDTH - 2*w
                        y_limit = self.WINDOW_HEIGHT - h

                        for x in range(0, x_limit+1):
                            for y in range(0, y_limit+1):
                                self.features.append([h_type, x, y, w, h])

                    if h_type == "HAAR_TYPE_III":
                        x_limit = self.WINDOW_WIDTH - 3*w
                        y_limit = self.WINDOW_HEIGHT - h

                        for x in range(0, x_limit+1):
                            for y in range(0, y_limit+1):
                                self.features.append([h_type, x, y, w, h])

                    if h_type == "HAAR_TYPE_IV":
                        x_limit = self.WINDOW_WIDTH - w
                        y_limit = self.WINDOW_HEIGHT - 3*h

                        for x in range(0, x_limit+1):
                            for y in range(0, y_limit+1):
                                self.features.append([h_type, x, y, w, h])

                    if h_type == "HAAR_TYPE_V":
                        x_limit = self.WINDOW_WIDTH - 2*w
                        y_limit = self.WINDOW_HEIGHT - 2*h

                        for x in range(0, x_limit+1):
                            for y in range(0, y_limit+1):
                                self.features.append([h_type, x, y, w, h])

    def calImgFeatureVal(self, IntegralMat, mat):
        """
        :param IntegralMat: the integral value of the image
        :return: a list including values of all features
        """
        featureVal = np.zeros(len(self.features))

        for feature_index in range(len(self.features)):
            h_type, x, y, w, h = self.features[feature_index]

            # #normalization
            # sumVal        = sum(sum(mat[y:y+h, x:x+w]))
            # sqSumVal      = sum(sum(mat[y:y+h, x:x+w] ** 2))
            # meanVal       = sumVal   / (w * h)
            # sqMeanVal     = sqSumVal / (w * h)
            #
            # normFactorVal = np.sqrt(sqMeanVal - meanVal ** 2)
            # if normFactorVal == 0:
            #     normFactorVal = 1


            if h_type == "HAAR_TYPE_I":
                pos = self.getPixelValInIntegralMat(x, y, w, h, IntegralMat)
                neg = self.getPixelValInIntegralMat(x, y+h, w, h, IntegralMat)

                featureVal[feature_index] = (pos - neg) / (2 * w * h)
            elif h_type == "HAAR_TYPE_II":
                neg = self.getPixelValInIntegralMat(x,   y, w, h, IntegralMat)
                pos = self.getPixelValInIntegralMat(x+w, y, w, h, IntegralMat)

                featureVal[feature_index] = (pos - neg) / (2 * w * h)
            elif h_type == "HAAR_TYPE_III":
                neg1 = self.getPixelValInIntegralMat(x, y, w, h, IntegralMat)
                pos  = self.getPixelValInIntegralMat(x+w, y, w, h, IntegralMat)
                neg2 = self.getPixelValInIntegralMat(x+2*w, y, w, h, IntegralMat)

                featureVal[feature_index] = (2*pos - neg1 - neg2)/ (3 * w * h)

            elif h_type == "HAAR_TYPE_IV":
                neg1 = self.getPixelValInIntegralMat(x, y, w, h, IntegralMat)
                pos  = self.getPixelValInIntegralMat(x, y+h, w, h, IntegralMat)
                neg2 = self.getPixelValInIntegralMat(x, y+2*h, w, h, IntegralMat)

                featureVal[feature_index] = (2*pos - neg1 - neg2) / (3 * w * h)

            elif h_type == "HAAR_TYPE_V":
                neg1 = self.getPixelValInIntegralMat(x, y, w, h, IntegralMat)
                pos1 = self.getPixelValInIntegralMat(x+w, y, w, h, IntegralMat)
                pos2 = self.getPixelValInIntegralMat(x, y+h, w, h, IntegralMat)
                neg2 = self.getPixelValInIntegralMat(x+w, y+h, w, h, IntegralMat)

                featureVal[feature_index] = (pos1 + pos2 - neg1 - neg2) / (4 * w * h)


        return featureVal

    def getPixelValInIntegralMat(self, x, y, w, h, integralMat):
        """
        x,y are the coordinates in the image matrix
        :param integralMat:
        :return:
        """
        if x == 0 and y == 0:
            return integralMat[y+h-1][x+w-1]
        elif x == 0:
            return integralMat[y+h-1][x+w-1] - integralMat[y-1][x+w-1]
        elif y == 0:
            return integralMat[y+h-1][x+w-1] - integralMat[y+h-1][x-1]
        else:
            return integralMat[y+h-1][x+w-1] + integralMat[y-1][x-1] \
                -  integralMat[y-1][x+w-1]   - integralMat[y+h-1][x-1]
