# README

This project consists of several Python scripts that allow you to preprocess your tweet dataset, train and evaluate models (including Support Vector Machines and Random Forests), and visualize results (e.g., class distribution and confusion matrices). Below is a step-by-step guide on how to run each part of the code, along with information about the required libraries.

---

## 1. Project Structure

Here is a brief overview of the files in this project:

1. `Prep.py`  
   - Reads an original dataset (e.g., `Tweets.csv`) and preprocesses it by:  
     • Filtering rows by specific negative reason categories.  
     • Removing user mentions.  
     • Balancing categories by undersampling.  
   - Outputs a cleaned, balanced CSV file named `tweets_preprocessed.csv`.  

2. `visualize.py`  
   - Reads `tweets_preprocessed.csv`.  
   - Plots the distribution of negative reasons (counts and proportions).  
   - Displays the plots in a simple side-by-side layout.  

3. `cross_validate_svm.py`  
   - Performs 10-fold cross-validation using TF-IDF + SVM.  
   - Outputs performance metrics (accuracy, precision, recall, F1) in a CSV file.  
   - Aggregates confusion matrices across all folds and plots a combined confusion matrix image.  

4. `cross_validate_rf.py`  
   - Similar to the above, but uses TF-IDF + Random Forest instead of SVM.  
   - Outputs performance metrics and a combined confusion matrix.  

---

## 2. Environment Setup

You need a Python environment (3.7 or later is recommended) with the following libraries installed:

- pandas  
- numpy  
- matplotlib  
- scikit-learn  

If you’re using Anaconda or Miniconda, you can install these libraries by running:

```bash
conda install pandas numpy matplotlib scikit-learn
```
Alternatively, if you are using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```
## 3. Running the Scripts

### A. Data Preprocessing
Run `prep.py`to generate `tweets_preprocessed.csv`
```bash
python Prep.py
```
This script will:

- Load the raw dataset
- Filter unwanted categories, remove user mentions, and balance the data.
- Create `tweets_preprocessed.csv`
### B. Visualization
Once `tweets_preprocessed.csv`is created,you can quickly visualize the distribution of categories by running `visualize.py`
```bash
python Visualize.py
```
This will:

- Read `tweets_preprocessed.csv`
- Plot the counts and proportions of each category in bar plots
- Display the plots
### C. Cross-Validation
Run `cross_validate_svm.py`or`cross_validate_rf.py`
```bash
python cross_validate_svm.py
```
or
```bash
pythpn cross_validate_rf.py
```
Steps performed:

- Read `tweets_preprocessed.csv`
- Perform 10-fold cross-validation using TF-IDF + SVM/RF
- Print metrics (accuracy, precision, recall, F1) for each fold
- Save a CSV file with the fold metrics and their average
- Aggregate the confusion matrix from all folds and save it as a PNG file


