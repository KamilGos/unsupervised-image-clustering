import keras
import time
from sklearn.decomposition import PCA
import pickle

class AutoEncoderAndPCA:

    def VGGPCAAutoEncoder(self, X_train, shape, LEARN, dir):
        print("### Feature extraction ###")
        start = time.time()
        vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet",
                                                     input_shape=tuple(shape))
        vgg16_output = self.VGGExtraction(vgg16_model, X_train)
        end = time.time()
        print("##### Feature extraction using CNN took {} seconds".format(round(end - start), 3))
        print("##### Flattened output has {} features".format(vgg16_output.shape[1]))

        start = time.time()

        if LEARN:
            vgg16_output_pca = self.PCAExtraction(vgg16_output, dir)
            print("##### Flattened output has {} features".format(vgg16_output_pca.shape[1]))

        end = time.time()
        print("##### Feature extraction using PCA (from CNN) took {} seconds".format(round(end - start), 3))

        if LEARN:
            return vgg16_output, vgg16_output_pca
        else:
            return vgg16_output

    @staticmethod
    def VGGExtraction(model, data):
        output = model.predict(data)
        output = output.reshape(data.shape[0], -1)
        return output

    @staticmethod
    def PCAExtraction(data, dir):
        pca = PCA(n_components=900)
        pca.fit(data)
        pickle.dump(pca, open((dir+'/pca_model.sav'), 'wb'))
        output = pca.transform(data)
        return output

    @staticmethod
    def PCAExtractionNew(data, dir):
        pca = pickle.load(open((dir+'/pca_model.sav'), 'rb'))
        output = pca.transform(data)
        return output

