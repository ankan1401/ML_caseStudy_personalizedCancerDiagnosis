# A Machine Learning case-study to classify genetic mutations based on evidence from clinical literature

### Business Objective:
  We have data for gentiec variation and mutation along with text based clinical literature. There are nine classes and we have to classify our data into one of these classes. This task we achieve on basis of the probabilites assigned to each of the class by our model. Since this is a critical problem, along with probabilites we also give the information that out of a given number of top features, how many of them are present in a test data point.
  
### Machine learning problem type:
  Since we have nine classess, this is a multiclass classification problem.

### Data Source:
  https://www.kaggle.com/c/msk-redefining-cancer-treatment/data

### Data Information:
  We have two data files.
  File 1 contents:
    Training variants (Id,Gene,Varaition,Class)
    Training text (Id,Clinical litreature text)
    
### Real-world/Business objectives and constraints:
  * No low-latency requirement.
  * Interpretability is important.
  * Errors can be very costly.
  * Probability of a data-point belonging to each class is needed.


### Machine learning objectives and constraints:
  * Penalize the errors : Log Loss
  * Interpretability
  * Class probabilities are needed
  * No latency requirement
  
### Performance Metric:
  * Log loss
  * Confusion matrix : Along with confusion matrix, we also use precision and recall matrix. These give us a better understanding
                       of how our model is working. 
  
### Approach to solve the problem:
  We will use a number of models here to see which model has the best result. 
  Models Used:
  * Naive Bayes
  * KNN
  * Logistic Regression with class balancing
  * Logistic Regression without class balancing
  * Random Forest
  * Linear SVM
  * Stacking classifier (Ensemble model)
  * Majority voting (Ensemble model)
 
### Data Preprocessing:
  * Remove all the stop words
  * Convert the text to lower case
  * Replace multiple spaces with a single space
  
### Data Featurization:
  We use two approaches for featurizing data:
  * Response coding: Better for Random forest
  * One hot encoding: Better for Logistic Regression
  * After vectorization, we stack all the three features to get the complete feature set for each data point.
  
### Train, test, cv split:
  * We split our whole data into training set, test set, cross validation set.
  * As it is a multiclass problem, we check that the distribution of classess is consistent through the train, test and cv set.
  
### Exploratory Data Analysis:
  We tried building a model on each of the individual features to ascertain wheter the feature is stable or not.
  If the test and cv error and not very much different from train error, we conculded that the feature is stable.

### Applying The Models:
  Once we have done the EDA and concluded that we will use all the features, we will now apply our ML models.
  We use Naive Bayes as our base line model, as it works well on text data.
  #### Steps:
  * Perform hyper parameter tunning to get the best values for the hyper parameters for each model
  * Use that hyper parameter to train the model and predict the results
