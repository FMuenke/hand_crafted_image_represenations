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

Open the train_image_classifier.py file and set the configurations as wanted. Run the training command and specify the location of the data set and the location of the model once it is trained.

````bash
python train_image_classifier.py -df PATH_TO_DATA_SET_TRAIN --model_folder PATH_TO_SAVE_MODEL_TO
````

If in doubt choose the following configurations for classification:

````python
opt = {
      "data_split_mode": "random",
      "classifier_opt": {
          "aggregator": "bag_of_words",
          "complexity": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
          "type": "random_forrest",
          "n_estimators": 5000,
      },
      "feature": ["hsv-kaze],
      "sampling_method": "kaze",
      "sampling_step": 0,
      "sampling_window": 0,
      "image_size": {
          "width": 256,
          "height": 256,
      },
  }
````
