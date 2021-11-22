import os
import numpy as np
import cv2
import copy

class ImageProcessing:
    def __init__(self, shape):
        self.images = []
        self.labels = []
        self.filenames = []
        self.images_norm = []
        self.labels_norm = []
        self.shape = tuple([shape[0], shape[1]])

    def loadImages(self, dir_name):
        for dirname, _, filenames in os.walk(dir_name):
            for filename in filenames:
                self.filenames.append(filename)
                self.labels.append((os.path.basename(dirname)))
                image = cv2.imread(os.path.join(dirname, filename))
                image = cv2.resize(image, self.shape, interpolation=cv2.INTER_LANCZOS4)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.images.append(image)
        print("### ", len(self.images), "images loaded")

    def normaliseImages(self):
        images = np.array(self.images, dtype=np.float32)
        labels = np.array(self.labels)
        images = images/255
        self.images_norm = copy.deepcopy(images)
        self.labels_norm = copy.deepcopy(labels)
        print("### Data shape: ", self.images_norm.shape)

    def returnData(self):
        return self.images_norm, self.labels_norm

    def returnFilenames(self):
        return self.filenames

if __name__ == "__main__":
    ImgProc = ImageProcessing((128, 128))
    ImgProc.loadImages("./test_images")
    ImgProc.normaliseImages()

