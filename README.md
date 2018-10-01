# A Machine Learning case-study to classify genetic mutations based on evidence from clinical literature

<h2> Business Objective: </h2>
<p> Given a {gene, variation, text-based clinical literature}, classify the data into one of the 9 classes.
 <br><b>Note: </b> There is no strict latency requirement but because of the critical nature of the problem, interpretability is very important. We need to specify the probability of a data point belonging to each class

<h2> Data Collection: </h2>
<p>
We get our data from :  https://www.kaggle.com/c/msk-redefining-cancer-treatment/data
</p>

<h2> Mapping to ML problem: Data</h2>
<p>We have two data files: one conatins the information about the genetic mutations and the other contains the clinical evidence (text) that human experts/pathologists use to classify the genetic mutations. Both these data files are have a common column called ID.<br>
Data file's information:
<ol>training_variants (ID , Gene, Variations, Class)</ol>
<ol>training_text (ID, Text)</ol>
</p>

<h2> Mapping to ML problem: Performance Metrices </h2>
<p>The problem here is a multi-class classification problem. Taking into consideration the problem type and the business constraints, we boil down to the following performance metrices:
<ol><b>Multi-class log-loss:</b> In case of probability scores, log-loss is one of the best performance metrices</ol>
<ol><b>Confusion Matrix:</b> Along with confusion matrix, we also use precision and recall matrix. These give us a better understanding
                       of how our model is performing.</ol>
</p>

<h2> Mapping to ML problem: Train-Test split </h2>
<p>The data doesn't have a temporal nature. So, we split the dataset randomly into 3 parts - Train(64%), CV(16%), Test(20%)
</p>

<h2> EDA: Preprocessing of text data</h2>
<p> We perform some basic pre-processing steps like removing NAN records, for each text record - replace every special character with space, convert all the characters into lower case and remove stop words. <br>
  We also plot the distribution of class labels in train, cv and test datasets and found that the distributions are similar
</p>

<h2> EDA: Prediction using a Random Model</h2>
<p> We simulate a random model where we generate the nine class probabilities such that they sum to 1. Now we compute the log-loss, basically to understand how bad our model can be. Any sensible model will have a log-loss less than random model's. We also build the confusion, precision and recall matrix to better understand how our random model is actually performing.<br>
  The log-loss came out to be roughly ~2.5 for both cv and test datasets
</p>

<h2> EDA: Univariate Analysis</h2>
<p> We performed univariate analysis for gene, variation and text feature and found them to be useful in one way or the other. Basically, we are not discarding any of the features.<br>
 Both Gene and Variation are categorical random variables. We featurize them using both OneHotEncoding and ResponseCoding. We choose the appropriate featurization based on the ML models we use. For converting text to vector, we can use CountVectorizer or Tf-Idf Vectorizer.
</p>

<h2> Data Preparation:</h2>
<p> We stack the three vectorized features and build our models on them(basically perform multivariate analysis). We will use the both the one-hot encoded and response coded features. Some models perform better with OHE and some with response coded
</p>

<h2> Modeling:Baseline Model - NaiveBayes</h2>
<p> Naive Bayes works very well on text data(high-dim data), so it often acts as a good baseline model. We use MultinomialNB in this case and a CalibratedClassifier on top of it. The log-loss on test came out around ~1.21. WE already have each class's probabilities for a data point and thus we can easily get FI for our NB model
</p>

<h2> Modeling:Other Models Applied</h2>
<p> We try out a few other models and some ensembles to see which model gives us the best result.<br> 
  Models Used:
  <ol>Naive Bayes</ol>
  <ol>KNN</ol>
  <ol>Logistic Regression with class balancing</ol>
  <ol>Logistic Regression without class balancing</ol>
  <ol>Random Forest</ol>
  <ol>Linear SVM</ol>
  <ol>Stacking classifier (Ensemble model)</ol>
  <ol>Majority Voting classifier (Ensemble model)</ol>
</p>

<h2> Conclusion</h2>
<p> Logistic Regression(with class-balancing) was our best model and performing some neat tricks with the features, we were able to obtain a log-loss less than 1
</p>
