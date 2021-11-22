Project: Unsupervised Classification
Author: Kamil Goś

Python version 3.5 or higher necessary

---- Necessary Libraries ---
Libraries delivered with Python:
os              # Miscellaneous operating system interfaces.
random          # Random number generation
time            # time measurements
pickle          # saving/loading models

Others:

matplotlib      # Ploting
Version: 3.1.3
Donwload: pip install matplotlib

sklearn         # machine learning
Version: 0.0
Download: pip install sklearn

prettytable     # creates table to print
Version: 0.7.2
Download: pip install PrettyTable

tensorflow      # neural networks
Version 2.0.0   !!!!
pip install -Iv tensorflow==2.0.0

Keras           # neural networks
Version 2.3.1
Download pip install keras

argparse        # argument parser
Version: 1.1
Download: pip install argparse

opencv          # image processing
Version: 4.2.0.32
Download: pip install opencv-python

pandas          # data storeing
Version: 1.0.4
Download: pip install pandas

seaborn         # Statistical data visualization
Version: 0.10.1
Download: pip install seaborn

--- HOW TO RUN ---

usage: main.py [-h] [-r] [-rt TRAIN_SET] [-rm RERUN_MODELS_DIR] [-c]
               [-ci IMAGES_TO_CLASSIFY] [-cm MODELS_DIR]

Unsupervised image classification. Use -h for more informations.

optional arguments:
  -h, --help            show this help message and exit
  -r, --rerun           Regenerate the classifiers
  -rt TRAIN_SET, --train_set TRAIN_SET
                        Directory with train set
  -rm RERUN_MODELS_DIR, --rerun_models_dir RERUN_MODELS_DIR
                        Directory to save new models
  -c, --classify        Classify images given in directory
  -ci IMAGES_TO_CLASSIFY, --images_to_classify IMAGES_TO_CLASSIFY
                        Directory with images to classify
  -cm MODELS_DIR, --models_dir MODELS_DIR
                        Directory with models which should be used to classify
                        new images


--- EXAMPLE OF USAGE ---
Classification of new images stored in "test_images" folder, using delivered models stored in "models" directory.
Images should be in .jpg format (extension). Not tested with other formats!

python main.py -c -ci ./test_images -cm ./models

Building and saving new models using train data stored in "raw_images" folder. Images can (but doesnt have to) be grouped
in nested directores (if so, they will get labels as directory name). Save output models in directory "modelsv2".

python main.py -r -rt ./raw_images -rm ./modelsv2



