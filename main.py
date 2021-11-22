from sources.preprocessing import ImageProcessing
from sources.feature_extraction import AutoEncoderAndPCA
from sources.clustering import ClustersModels
from argparse import ArgumentParser
import os

dir = {'rgb': 0, 'grey': 1, 'canny': 2, 'thresh': 3}


def learnNewModels(in_images_dir, out_models_dir):
    ImgProc = ImageProcessing([224, 224, 3])
    ImgProc.loadImages(in_images_dir)
    ImgProc.normaliseImages()
    data = ImgProc.returnData()

    Extractor = AutoEncoderAndPCA()
    features, features_pca = Extractor.VGGPCAAutoEncoder(data[0], [224, 224, 3], True, out_models_dir)

    Clusters = ClustersModels()
    Clusters.trainModels(out_models_dir, features, features_pca, data)


def predictNewData(in_images_dir, in_models_dir):
    ImgProc = ImageProcessing([224, 224, 3])
    ImgProc.loadImages(in_images_dir)
    ImgProc.normaliseImages()
    data = ImgProc.returnData()

    Extractor = AutoEncoderAndPCA()
    features = Extractor.VGGPCAAutoEncoder(data[0], [224, 224, 3], False, None)

    Clusters = ClustersModels()
    Clusters.predictNewData(in_models_dir, features, ImgProc.returnFilenames())


if __name__ == "__main__":

    parser = ArgumentParser(description='Unsupervised image classification. Use -h for more informations.')

    parser.add_argument('-r', '--rerun', action="store_true", help="Regenerate the classifiers")
    parser.add_argument('-rt', '--train_set', help="Directory with train set", required=False)
    parser.add_argument('-rm', '--rerun_models_dir', help="Directory to save new models", required=False)
    parser.add_argument('-c', '--classify', action="store_true", help="Classify images given in directory")
    parser.add_argument('-ci', '--images_to_classify', help="Directory with images to classify", required=False)
    parser.add_argument('-cm', '--models_dir', help="Directory with models which should be used to classify new images",
                        required=False)
    args = parser.parse_args()

    print("*********** PARSER INFO **************")
    if args.classify:
        print("Directory with images to classification: ", args.images_to_classify)
        print("Directory with models which should be used to classification : ", args.models_dir)
    if args.rerun:
        print("Directory with train set: ", args.train_set)
        print("Directory where new models should be saved: ", args.rerun_models_dir)

    if not args.classify and not args.rerun:
        print("No options selected. Use -c or -r. For help use -h")


    if args.classify or args.rerun:
        if args.rerun:
            train_data_dir = str(args.train_set).replace("'", "")
            train_data_dir = train_data_dir.replace('"', '')
            if not os.path.exists(train_data_dir):
                print(train_data_dir, " does not exist !")
            else:
                out_models_dir = str(args.rerun_models_dir).replace("'", "")
                out_models_dir = out_models_dir.replace('"', '')
                if not os.path.exists(out_models_dir):
                    os.makedirs(out_models_dir)
                print("+++ RUNNING MODELS GENERATION +++")
                learnNewModels(train_data_dir, out_models_dir)

        if args.classify:
            in_images_dir = str(args.images_to_classify).replace("'", "")
            in_images_dir = in_images_dir.replace('"', '')
            models_dir = str(args.models_dir)
            models_dir = models_dir.replace('"', '')

            if not os.path.exists(in_images_dir):
                print(in_images_dir, " does not exist !")
            elif not os.path.exists(models_dir):
                print(models_dir, " does not exist !")
            else:
                print("+++ RUNNING CLASSIFICATION +++")
                predictNewData(in_images_dir, models_dir)

