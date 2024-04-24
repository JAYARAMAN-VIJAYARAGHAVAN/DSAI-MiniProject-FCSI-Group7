# DSAI-MiniProject-FCSI-Group7
Predicting Lung Cancer Risk Using Logistic Regression
Project Overview:
This project aims to predict the risk of lung cancer using logistic regression. The dataset used here contains various attributes related to individuals' health, behaviours, and demographics. By analysing these attributes, we can predict whether an individual is at risk of lung cancer or not. We employ logistic regression, a popular algorithm for binary classification problems.
Dataset:
The dataset used for this project is "survey_lung_cancer.csv". It contains the following columns:
•	AGE
•	GENDER
•	SMOKING
•	YELLOW_FINGERS
•	ANXIETY
•	PEER_PRESSURE
•	CHRONIC_DISEASE
•	FATIGUE
•	ALLERGY
•	WHEEZING
•	ALCOHOL_CONSUMING
•	COUGHING
•	SHORTNESS_OF_BREATH
•	SWALLOWING_DIFFICULTY
•	CHEST_PAIN
•	LUNG_CANCER (Target Variable).
Project Workflow:
1.	Data Exploration and Preprocessing:
•	Import the necessary libraries: NumPy, Pandas, Seaborn, and Matplotlib.
•	Load the dataset and display its first few rows using pandas.DataFrame.head() method.
•	Display the information about the dataset using pandas.DataFrame.info() method.
•	Check for and remove any duplicate records using pandas.DataFrame.drop_duplicates() method.
•	Check for missing values using pandas.DataFrame.isnull().sum().
•	Visualize the distribution of the target variable using Seaborn's catplot() method.
2.	Data Balancing:
•	Address the class imbalance issue by upsampling the minority class (individuals with no lung cancer) using sklearn.utils.resample() method.
3.	Data Preprocessing:
•	Convert categorical variables into numerical using Label Encoding from sklearn.preprocessing.
•	Encode the categorical variables: 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', and 'LUNG_CANCER'.
4.	Exploratory Data Analysis (EDA):
•	Visualize the relationship between various features and the target variable using Seaborn's stripplot() method.
•	Perform feature selection using chi-squared test.
5.	Feature Selection:
•	Use chi-squared test to select the most important features for predicting lung cancer risk.
6.	Model Development:
•	Split the data into training and testing sets using sklearn.model_selection.train_test_split() method.
•	Train a logistic regression model using sklearn.linear_model.LogisticRegression().
7.	Model Evaluation:
•	Evaluate the model's performance using accuracy score.
•	Visualize the confusion matrix using Seaborn's heatmap() method.
Conclusion:
This project successfully predicts the risk of lung cancer using logistic regression. By employing various attributes related to health, behaviours, and demographics, the model provides a valuable tool for identifying individuals at risk of lung cancer. The project demonstrates the implementation of a binary classification problem using a logistic regression algorithm. Through proper data preprocessing, feature selection, model development, and evaluation, it provides insights into building a predictive model for lung cancer risk assessment. Multiple models were made and the best one out of them was chosen.

Contributions:

Vijay: Data preparation, exploratory analysis

Avneesh: Model Training, Slides

References:

M2ExploratoryAnalysis.ipnyb from NTUlearn
https://www.cancer.org/cancer/types/lung-cancer/detection-diagnosis-staging/detection.html#:~:text=If%20lung%20cancer%20is%20found,have%20any%20signs%20or%20symptoms
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7950268/

