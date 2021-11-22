
<!-- image -->
<div align="center" id="top"> 
  <img src="https://data-flair.training/blogs/wp-content/uploads/sites/2/2019/07/unsupervised-learning.png" width="500" />
  &#xa0;
</div>

<h1 align="center"> unsupervised-image-clustering </h1>
<h2 align="center"> Apply K-means and Agglomerative Clustering algorithms to the images from a given datase </h2>

<!-- https://shields.io/ -->
<p align="center">
  <img alt="Top language" src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge&logo=python">
  <img alt="Status" src="https://img.shields.io/badge/Status-done-green?style=for-the-badge">
  <img alt="Code size" src="https://img.shields.io/github/languages/code-size/KamilGos/unsupervised-image-clustering?style=for-the-badge">
</p>


<!-- table of contents -->
<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0;
  <a href="#package-content">Content</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0; | &#xa0;
  <a href="#eyes-implementation">Implementation</a> &#xa0; | &#xa0;
  <a href="#microscope-tests">Tests</a> &#xa0; | &#xa0;
  <a href="#memo-license">License</a> &#xa0; | &#xa0;
  <a href="#technologist-author">Author</a> &#xa0; | &#xa0;
</p>

<br>

## :dart: About ##
The project aimed to apply two clustering algorithms (K-means and Agglomerative Clustering) to the images from a given dataset.

## :package: Content
 * [main.py](main.py) - executable file
 * [sources](sources) - directory with source files
 * [data/learn_images](data/learn_images) - directory with learn dataset
 * [data/test_images](data/test_images) - direcory with test test dataset

## :checkered_flag: Starting ##
```bash
# Clone this project
$ git clone https://github.com/KamilGos/unsupervised-image-clustering

# Access
$ cd unsupervised-image-clustering

# Run the project
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

$ python main.py -c -ci ./test_images -cm ./models

Building and saving new models using train data stored in "raw_images" folder. Images can (but doesnt have to) be grouped
in nested directores (if so, they will get labels as directory name). Save output models in directory "modelsv2".

$ python main.py -r -rt ./raw_images -rm ./modelsv2
```

## :exclamation: Requirements ##

* tensorflow Version 2.0.0   
`pip install -Iv tensorflow==2.0.0`
* Keras Version 2.3.1   
`pip install -Iv keras==2.3.1`

## :eyes: Implementation ##
<h2>Preliminary analysis and preprocessing</h2>
The dataset contains 1000 of images of 10 different categories: motorcycles, trees, chandeliers, watches, tigers, planes, grand pianos, cars, turtles and boats. Every category contains exactly 100 images. All images are represented in RGB colour space (instead of cars which are in greyscale) and they have a variety of sizes. Figure below shows the random image for every category. As we have to implement unsupervised learning the labels of images won't be use during training the models. They will be use just for testing. 

<br>


<div align="center" id="put_id"> 
  <img src=images/rand_sample.png width="500" />
  &#xa0;
</div>

<br>

Before applying our data to feature extraction algorithms it is necessary to convert them toproper form. All images must have the same size. As a target dimension 224x224 size wasselected.

Before applying images to segmentation algorithms the data should be reduced. The very good approach to reduce the data size is **feature extraction**. Feature extraction is a process of dimensionality reduction by which an initial set of raw data is reduced to more manageable groups for further processing. One of the approaches to feature extraction are autoencoders. 

Since our inputs are images in RGB colour space, it makes sense to use **convolutional neural networks (convnets or CNN)** as encoders, they simply perform much better. Therefore, the popular VGG16 CNN was used during the experiment. It is considered to be one of the excellent vision model architecture till date. During the experiments, the network without 3 fully-connected layers at the top of the network was used. It means that the output of the VGG16 net is 25088 length vector. The default input of this neural network is 224x224x3 matrix (image). With this input size, the chosen CNN work best. That is why our images were reshaped to this size. 

25088 of features per image is still quite a large number. To decrease this the Principal component analysis (PCA) was used. Principal component analysis (PCA) is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance. Finding such new variables, the principal components, reduces to solving an eigenvalue/eigenvector problem, and the new variables are defined by the dataset at hand, not a priori, hence making PCA an adaptive data analysis technique.

Table below shows the data reduction flow (from left to right). Encoding features using CNN is a time-consuming process. For our data, it took 153 seconds. Extracting features from VGG16 output using PCA took 6 seconds. Data in this format is ready to be loaded into clustering algorithms.


<div align="center" id="put_id"> 
  <img src=images/tab1.png width="400" />
  &#xa0;
</div>


## :microscope: Tests ##
To have a better comparison, both straight from VGG16 features and those generated fromPCA were used during the model training.

Table below shows the scores for all methods. As we can see the Agglomerative Clustering algorithm works much better. Also, the PCA algorithm gives a very high time reduction without losing accuracy. From confusion matrices, we can observe that the K-means algorithm had a huge problem to distinguish the tree from the chandelier while Agglomerative Clustering algorithm did it very well but the training took him almost 3 times more time, but the Agglomerative Clustering iv very specific. The problem is that it is not suitable for classifying new data that is in a much smaller amount or that all classes that were used during learning do not appear in the database. The conclusion is that we can classify the images using unsupervised learning, but still the supervised learning methods are better.

<div align="center" id="put_id"> 
  <img src=images/tab2.png width="500" />
  &#xa0;
</div>

<br>

<h2>Confusion matrices</h2>

<div align="center">

| Algorithm   | Confusion Matrix    |
|--------------- | --------------- |
| K-Means  | <img src=images/kmeans.png width="500" />  |
| K-Means + PCA   | <img src=images/kmeans_pca.png width="500" />   |
| AGG   | <img src=images/agg.png width="500" />   |
| AGG + PCA   | <img src=images/agg_pca.png width="500" />   |

</div>

## :memo: License ##

This project is under license from MIT.

## :technologist: Author ##

Made with :heart: by <a href="https://github.com/KamilGos" target="_blank">Kamil Goś</a>

&#xa0;

<a href="#top">Back to top</a>
