**Leaf Recognition and Classification system using Shape and Colour features with Random Forest Classifiers**

To use the Leaf Classification system, download and extract the zip file of this project in the same directory as your dataset.
You can download the dataset from following link:
(https://archive.ics.uci.edu/ml/datasets/Folio)

**NOTE**
Before running any of the files below, make sure you change the path to the Folio Leaf Dataset directory is added correctly(varies for each system).
**Windows Users**: Change variable **t="\"** to **t="//"** in each code file.

Then run the files in **"code"** directory in following sequence:
1. **Geometric_features.py**
2. **Histogram.py**
3. **HOG.py**
4. **Post Processing.ipynb** [You'll need Jupyter Notebook to run this file.]
5. (Optional) **Predictor.py** [If you want to predict a single leaf]


* **"doc_img"** directory contains sample images generated while pre-processing and feature extraction.
* **"csv"** directory contains the csv files generated during feature extraction
* **"trained_classification_model"** directory contains the trained Random Forest Classifier Model on the Folio Leaf Dataset

* **Plants list.pdf** contains the description of the Folio Dataset
