# Practical Application III: Comparing Classifiers

 
### Problem 1: Understanding the requirements and Data

To gain a better understanding of the data, I looked at information provided in the CRISP-DM-BANK.PDF, and examine the **Materials and Methods** section of the paper. 

 1. In order to stay afloat in the face of intense industry demands and financial uncertainty, banks often develop strategies such as offering investments with attractive interest rates. To maximize success while minimizing effort, they then focus their efforts on targeted marketing campaigns designed to produce positive results - like attracting customers for long-term deposits.

 2. he data used for this excerise are from a Portuguese bank that used its own contact-center to do directed marketing campaigns. The telephone, with a human agent as the interlocutor, was the dominant marketing channel, although sometimes with an auxiliary use of the Internet online banking channel (e.g. by showing information to specific targeted client). Furthermore, each campaign was managed in an integrated fashion and the results for all channels were outputted together.

 3. The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts. During these phone campaigns, an attractive long-term deposit application, with good interest rates, was offered. For each contact, a large number of attributes was stored and if there was a success (the target variable). There were 6499 successes (8% success rate).

 **Objective**: Our goal is to use powerful Machine Learning models, such as K Nearest Neighbor, Logistic Regression, Decision Trees and Support Vector Machines against our dataset in order to reduce marketing campaign costs. Furthermore, we'll be comparing the performance of these models with each other and providing a tool for filtering datasets which will help increase future campaigns' success rates!

### Problem 2: Read in the Data

Use pandas to read in the dataset `bank-additional-full.csv` 

 1.  the data set has 41188 entries, 0 to 41187 and total of 21 columns
 
### Problem 3: Understanding the Features

Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.

## Input variables:
# bank client data:
1. - age (numeric)
2. - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. - default: has credit in default? (categorical: 'no','yes','unknown')
6. - housing: has housing loan? (categorical: 'no','yes','unknown')
7. - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8. - contact: contact communication type (categorical: 'cellular','telephone')
9. - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12. - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. - previous: number of contacts performed before this campaign and for this client (numeric)
15. - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16. - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17. - cons.price.idx: consumer price index - monthly indicator (numeric)
18. - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. - euribor3m: euribor 3 month rate - daily indicator (numeric)
20. - nr.employed: number of employees - quarterly indicator (numeric)

## Output variable (desired target):
21. - y - has the client subscribed a term deposit? (binary: 'yes','no')

Our Examine uncovered no missing values but we must apply convert_dtypes() to ensure our columns have the appropriate data type. Doing so will help us utilize our data more effectively!

### Problem 4: Understanding the Task

To meet the business objective of improving marketing campaign efficiency and maintaining a similar number of successful outcomes with fewer contacts, the following tasks should be carried out:

1. Clean and preprocess the data to handle missing values, inconsistencies, and categorical variables.
2. Perform feature engineering and selection to identify the most relevant features for modeling.
3. Split the data into training and testing sets to evaluate model performance on unseen data.
4. Select and train appropriate classification models, such as logistic regression, decision trees, or support vector machines.
5. Optimize model hyperparameters using techniques like cross-validation, grid search, or randomized search.
6. Evaluate and compare model performance using relevant metrics to select the best model(s) for deployment.
7. Interpret the selected model(s) to understand important features and refine marketing strategies.
8. Deploy the model(s) in a production environment to support decision-making and optimize marketing campaigns.
9. Monitor and maintain the deployed model(s) to ensure continued performance in line with business objectives.

By performing these tasks, you can work towards enhancing the efficiency of marketing campaigns and achieving a similar number of successful customer subscriptions with reduced contact efforts.

### Problem 5: Engineering Features

Now that you understand your business objective, we will build a basic model to get started.  Before we can do this, we must work to encode the data.  Using just the bank information features (columns 1 - 7), prepare the features and target column for modeling with appropriate encoding and transformations.

Perform statistical tests to determine the association between each categorical feature and the target variable. I haved used the Chi-squared test to assess the independence between a categorical feature and the target variable. A low p-value from the Chi-squared test suggests that the categorical feature and the target variable are dependent, which could make the feature useful for your model.

Categorical features: ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
job - p-value: 4.189763287563623e-199
marital - p-value: 2.068014648442211e-26
education - p-value: 3.3051890144025054e-38
default - p-value: 5.161957951391637e-89
housing - p-value: 0.05829447669453452
loan - p-value: 0.5786752870441754
contact - p-value: 1.525985652312996e-189
month - p-value: 0.0
day_of_week - p-value: 2.958482005278532e-05
poutcome - p-value: 0.0
y - p-value: 0.0
Selected Categorical features: ['housing', 'loan']

**Based on data we will drop features: ['housing', 'loan'] that has no relevnes due to soce grter or equil than 0.05**

Based on the visualization of relationships between each feature and the target variable, the plots reveal that the following columns may not have a significant effect on the machine learning model:
day_of_week and pdays,

we will drop **day_of_week and pdayshem** to increase processing te performance**

1. Encode categorical features using LabelEncode
2. Separate features and target variable
3. Train a random forest classifier

**Display feature importances**

 Index     feature  importance.  
7         duration    0.350923.  
14       euribor3m    0.129292.  
0              age    0.107838.  
15     nr.employed    0.077533.  
1              job    0.053862.  
8         campaign    0.048115.  
3        education    0.047974.  
10        poutcome    0.037750.  
2          marital    0.025998.  
13   cons.conf.idx    0.025640.  
12  cons.price.idx    0.021233.  
11    emp.var.rate    0.019662.  
6            month    0.017396.  
9         previous    0.016502.  
5          contact    0.010469.  
4          default    0.009813.  
  
### Problem 6: Train/Test Split

With our data prepared, split it into a train and test set

**Separate features and target variable**
X = data_encoded.drop('y', axis=1)
y = data_encoded['y']

**Split the dataset into training and testing sets**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

**Standardize or normalize the features**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### Problem 7: A Baseline Model

Before we build our first model, we want to establish a baseline.  What is the baseline performance that our classifier should aim to beat?

Based on our test data 
Baseline Accuracy (Majority Class Classifier  where customer will not excpet the offer is): 0.8876
Baseline Accuracy (Minority Class Classifier  where customer will  excpet the offer is): 0.1124

### Problem 8: A Simple Model

Use Logistic Regression to build a basic model on your data.  

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
[[10690   278] [  832   557]]  

### Problem 9: Score the Model

What is the accuracy of our logistic_regression model?

The accuracy of a model is a performance metric that measures the proportion of correct predictions made by the model out of the total number of predictions. In other words, it is the ratio of the number of correct predictions to the total number of predictions made. Accuracy is commonly used to evaluate the performance of classification models.

Accuracy: 0.9102
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.97      0.95     10968
           1       0.67      0.40      0.50      1389

    accuracy                           0.91     12357
   macro avg       0.80      0.69      0.73     12357
weighted avg       0.90      0.91      0.90     12357

Confusion Matrix:
[[10690   278] [  832   557]]


### Problem 10: Model Comparisons

Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models.  Using the default settings for each of the models, fit and score each.  Also, be sure to compare the fit time of each of the models.  Present your findings in a `DataFrame` similar to that below:
-------------------------------------------------------------------------------------------
                 Model | Train Time |  Accuracy |  Train Accuracy |  Test Accuracy|
                   SVM |    6.360792|  0.911791 |        0.918248 |     0.911791  |
   Logistic Regression |    0.039036|  0.910172 |       0.909091  |     0.910172  |
                   KNN |    0.002002|  0.900461 |       0.930041  |     0.900461  |
         Decision Tree |    0.107098|  0.888161 |       1.000000  |     0.888161  |
         ------------------------------------------------------------------------------

### Problem 11: Improving the Model

Now that we have some basic models on the board, we want to try to improve these.  Below, we list a few things to explore in this pursuit.

- More feature engineering and exploration.  For example, should we keep the gender feature?  Why or why not?
- Hyperparameter tuning and grid search.  All of our models have additional hyperparameters to tune and explore.  For example the number of neighbors in KNN or the maximum depth of a Decision Tree.  
- Adjust your performance metric

I Pweromed GridSearchCV and RandomizedSearchCV to hyperparameters to tune and explore amd look bets prametrs

        Model                 Train_Time  Best Parameters                                      Best Score
0      LogisticRegression()   12.304204   {'solver': 'liblinear', 'penalty': 'l2', 'C': ...    0.909264
1    KNeighborsClassifier()   33.202210   {'weights': 'distance', 'n_neighbors': 28, 'me...    0.906802
2  DecisionTreeClassifier()    1.506371   {'min_samples_split': 15, 'min_samples_leaf': ...    0.913774
3                     SVC()  684.002242   {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}    0.909334

                                       
From these resuts DecisionTreeClassifier() scoer was higest 0.913774 na d excution time of 1.506371
Comprare to result of KNeighborsClassifier()  where it took 33 and lowast  scoer of 0.906802
the winnr fo this teste is  Best Score() with 1.5 min and Best Score  0.913774

##### Questions

Is the procedd forn a bets ML Model? what is the reulsts

**DecisionTreeClassifier() score was highest 0.913774 and execution time of 1.506371. in most cases, this model would perform well.**



