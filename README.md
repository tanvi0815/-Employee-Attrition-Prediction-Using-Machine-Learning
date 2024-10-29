 Employee Attrition Prediction Using Machine Learning

Employee attrition refers to an employee's voluntary or involuntary departure from a workforce. Organizations invest significant resources in recruiting talented individuals and training them. 
Every employee plays a crucial role in a company's success. In my project, I focused on forecasting employee attrition and identifying the factors that lead to an employee leaving the organization. 
trained various classification models on my dataset and evaluated their performance using different metrics, including accuracy, precision, recall, and F1 Score.
Additionally, analyzed the dataset to identify the primary factors influencing employee turnover.
This project aims to provide organizations with valuable insights into the drivers of attrition, ultimately helping to improve retention rates.



**Machine Learning Models

trained and evaluated 9 supervised machine learning classification models.

1. Logistic Regression
2. Naive Bayes
3. Decision Tree
4. Random Forest
5. AdaBoost
6. Support Vector Machine
7. Linear Discriminant Analysis
8. Multilayer Perceptron
9. K-Nearest Neighbors

** Datasets
 trained our models on 6 different datasets
1. Imabalanced
2. Undersampled
3. Oversampled
4. PCA
5. Undersampling With PCA
6. Oversampling With PCA

Further, to get the best performance, hyperparameter tuning was carried out using RandomSearchCV and GridSearchCV. K-fold cross-validation with 5 folds was also
performed on the training set. To handle model interpretability, appropriate graphs and figures were used.Accuracy for the attrition decision is a biased metric, and hence  evaluated the model on all the
following classification metrics: accuracy, precision, recall
and F1 Score.

** Dataset
Used the [IBM Employee Attrition dataset from Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset). It contains 35 columns and 1470 rows and has a mix of
numerical and categorical features.

** Best Performing Model
The best performance was obtained in Random Forest Model
with PCA and Oversampling with an accuracy of 99.2%,
the precision of 98.6%, recall of 99.8% and F1 Score of
99.2%.

** Instructions to run
Jupyter Notebook can be run using Google Colab or locally using Anaconda Navigator.

**Steps to run using Google Colab
1. Upload the dataset
2. Click on Runtime -> Run all / Restart and Run all

** Libraries Used
1. [Numpy](https://numpy.org/)
2. [Pandas](https://pandas.pydata.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [Seaborn](https://seaborn.pydata.org/)
5. [Scikit-learn](https://scikit-learn.org/stable/index.html)
