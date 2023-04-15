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

**we will drop tday_of_week and pdayshem to increase processing te performance**

1. Encode categorical features using LabelEncode
2. Separate features and target variable
3. Train a random forest classifier

**Display feature importances**

           feature  importance
7         duration    0.350923
14       euribor3m    0.129292
0              age    0.107838
15     nr.employed    0.077533
1              job    0.053862
8         campaign    0.048115
3        education    0.047974
10        poutcome    0.037750
2          marital    0.025998
13   cons.conf.idx    0.025640
12  cons.price.idx    0.021233
11    emp.var.rate    0.019662
6            month    0.017396
9         previous    0.016502
5          contact    0.010469
4          default    0.009813
