# classic_image_classification

## Setup
Installation is done by cloning the repository
```bash
git clone https://github.com/FMuenke/classic_image_classification.git
```
All dependency can be installed with
````bash
pip install -r requirements.txt
````
It is strongly recommended to use a virtual environment (like anaconda).


## Data Set

The data set should be separated in train and test folder. In ech folder should be one "images" folder and one "labels" folder. With labels and images stored in the corresponding folder. The label file should have the same name as the image file (excluding the suffix). Each label file is a text file containing all classes present on each image (one class per line).


## Training

Open the train_image_classifier.py file and set the configurations as wanted. 
Run the training command and specify the location of the data set and the location of the model once it is trained.

````bash
python train_image_classifier.py -df PATH_TO_DATA_SET_TRAIN --model_folder PATH_TO_SAVE_MODEL_TO -clmp PATH_TO_CLASSMAPPING.JSON
````

## Example
The training can be further customized. As shown in the following code snippet:


````python
from classic_image_classification.machine_learning import BestOfBagOfWords

# The Best of Bag of Words Classifier is the most general classifier.
# It includes an automatic parameter search.

# 1. Define options

opt = {
    "data_split_mode": "random",                  # How to split data into train and test
    "aggregator": "bag_of_words",                 # Define aggregation ["bag_of_words", "vlad", ...]
    "complexity": [8, 16, 32, 64, 128, 256, 512], # Define the complexity (numer of clusters) of the aggregation
    "type": ["mlp", "mlp_x"],                     # Which classifier should be used [mlp, rf, svm, ...]
    "feature": ["hsv-sift", "gray-sift"],         # Which Features should be tested
    "sampling_method": "dense",                   # How to sample features
    "sampling_step": [16, 32],                    # Option for dense-sampling: step size
    "sampling_window": [16, 32],                  # Option for dense-sampling: window size
    "image_size": [                               # Resize the image (otherwise set to None)
        {
            "width": 128,
            "height": 128,
        }
    ]
}

# 2. Define Classes for the classifier

class_mapping = {
    "cls_1": 0,
    "cls_2": 1,
    # ...
}


MODEL_FOLDER="/Path/to/model"
PATH_TO_DATASET="/Path/to/folder/with/training/data"

bob = BestOfBagOfWords(opt, class_mapping)
bob.fit(MODEL_FOLDER, PATH_TO_DATASET)
````

The provided code generates a set of image classifiers and automatically selects the best one. 
To further evaluate and use the classifier, simply load it from the folder it was saved to.

````python
import os
from classic_image_classification.machine_learning import ClassicImageClassifier

MODEL_FOLDER="/Path/to/model"
PATH_TO_DATASET="/Path/to/folder/with/training/data"
PATH_TO_VISUALIZE_RESULTS=os.path.join(MODEL_FOLDER, "image_classification")

model = ClassicImageClassifier()
model.load(MODEL_FOLDER)
model.evaluate(PATH_TO_DATASET, PATH_TO_VISUALIZE_RESULTS)
````


## Options

#### Data Split (data_split_mode)
- random: Data is split completely randomized
- fixed: Data is split fixed (last X percent)

#### Type of Feature (feature)
Features are separated into color-space and feature-type: *COLOR-FEATURE*

###### Color Options: gray, hsv, opponent, rgb, RGB

###### Features:
- hog: Histogram of Oriented Gradients. Define options with: hog+SIZE+NORMALIZATION ("L1", "L1SQRT", "L2", "L2HYS") ("TINY", "SMALL", "BIG", "HUGE"). E.g. hog+SMALL+L2
- glcm: Gray Level Co-Matrix
- lbp: Local Binary Pattern. Define options with: lbp+NPOINTS+RADIUS. E.g. lbp+24+7
- lm: Leung Malik Filter Bank
- histogram: Histogram of Values. Define options with: histogram+NUMBINS. E.g. histogram+32
- OpenCV Descriptors: [kaze, akaze, sift, orb, brisk]


#### Sampling Method of Features (sampling_method)
- dense: Sample features with a grid: Define as well
  - sampling_step: Distance between Features
  - sampling_window: Size of window to compute the Feature from
- one: Compute the Feature from the full image
- OpenCV KeyPoint Detectors: [orb, akaze, brisk, sift]

#### Aggregator (aggregator)
Based on previous setting for each image are multiple feature vectors (descriptors) computed.
As representation for an image they need to be summarized (aggregated) into one new feature vector.

- bag_of_words: Summarize all descriptors as a bag of words
- fisher_vector: Summarize all descriptors as a Fisher Vector
- vlad: Summarize all descriptors with the VLAD algorithm
- global_avg: Compute the average of all descriptors
- global_max: Compute the maximum of all descriptors


#### Aggregator Complexity (complexity)
For some aggregation methods [bag_of_words, fisher_vector, vlad] the complexity can be set. 
It represents the amount of distinct clusters to summarize features.


#### Classifier
The classifier used to make the final classification
- lr: Logistic Regression
- svm: Support Vector Machine
- mlp: Multi-Layer-Perceptron, Neurons: (64)
- mlp_x: Multi-Layer-Perceptron, Neurons: (128, 64)
- mlp_xx: Multi-Layer-Perceptron, Neurons: (256, 128, 64)
- rf: Random Forrest
- knn_3: K-Nearest Neighbours (N=3)
- knn_5: K-Nearest Neighbours (N=5)
- knn_7: K-Nearest Neighbours (N=7)