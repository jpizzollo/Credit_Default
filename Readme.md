# Predicting Credit Card Default

This aim of this project is to predict when a customer will default on a credit card loan given some basic demographics and recent payment history. A random forest model was used to predict default on loans with an 81% accuracy. The major limitation in prediction is recall for class default. This is possibly due to the instances of customers who default on loans while carrying little debt. Additional information such as credit scores, longer payment history. or supplementary demographic data could help improve accuracy and recall in this model.

### Data Preparation

The UCI default of credit card clients Data Set was used to construct this model. Features include customer credit limit, gender, education, marital status, age, six months of repayment status, six months of bill statements, and six months payments. Random raining, validation, and testing splits (60%, 20%, and 20%, respectively) were maintained throughout model training and evaluation. Dummy variables were created with one-hot encoding, and scaled feature sets were prepared for use in KNN and SVC models. In this dataset, 22% of cases are classified as default. To balance classes, SMOTE was used to upsample default cases in the training set.

### Model Evaluation

Six basic models were prepared to evaluate for accuracy and recall. K-nearest neighbors, logistic regression, Gaussian naive Bayes, support vector classification, decision tree, and random forest were initially evaluated. The only hyperparameter tested prior was k-neighbors, otherwise default parameters were used.

Evaluating for accuracy, random forest, svc, and knn performed best with 0.81 - 0.82 accuracy scores. Of these, random forest showed the best recall at 0.4. Recall is important for this classification since it represents the number of missed default predictions (false negatives) - ie a higher risk scenario than false positives. Additionally, the ROC AUC was calculated for each model and the random forest had the highest score at 0.76.

A default threshold (0.5) for prediction with random forest resulted in an imbalance between precision and recall. To balance this and thus improve recall, precision and recall were calculated for probability thresholds between 0.01 and 1. At 0.34, recall was more balance with precision and increased to 0.53. Thus, the model captures 53% of default cases. Predicting non-default cases, the model captures 87% of these cases.

### Feature engineering

To simplify parameters that inform about credit default, a new set of features were created that describe current debt status. The original dataset has six months of bill and payment history, which can be simplified in terms of outstanding debt. These five new features were used to replace 18 original features: six months each of bill statements, payment amount, and repayment status for a cost of 0.04 on accuracy while allowing more interpretability and simplification for interactive visualization.

### Demonstrating model predictions

A web app was created to demonstrate model predictions given a set of imput parameters. This streamlit application loads the random forest model and raw data (for visualization) and uses a python script (streamlit_credit_app.py) to create an interface that allows a user to toggle categorical or numeric features and calculate probability of credit default given any set of parameters.
