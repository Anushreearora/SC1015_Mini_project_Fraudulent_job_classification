# SC1015_Mini_project_Fraudulent_job_classification

## Problem Motivation 
Recently, with the impact of Covid-19, causing a scarcity of jobs and the proliferation of remote work, many people are eagerly looking to find work and earn a livelihood. Many scammers are preying on this desperation and creating fraudulent job postings to scam their victims. Once they find their victims, they are easily able to gain access to variety of personal information including the bank details of their victims. Our objective, is to find a way to help these people differentiate between real and fake employment opportunities.

## The Dataset 
The dataset we will be using is called the recruitment scam data set, from Kaggle - https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction, that was uploaded by Shivam Bansal. This is a highly organised dataset with clearly defined variables. However, it contains a wide variety of data types and needs to be cleaned for any meaningful analysis. 

## Problem Formulation 
To build a classifier than can identify fraudulent and non-fraudulent jobs.

We will be using 2 machine learning models and comparing their effectiveness. The first model is a textual classification model that uses natural language processing to identify fraudulent postings. Our second model is a categorical classification model, that uses a random forest.

The effectiveness of the results of both models in identifying fraudulent job postings will be reviewed using accuracy metrics.

## Exploratory Data Analysis 

![non-fraudulent - fraudulent ratio](https://user-images.githubusercontent.com/104303593/164985904-8e7690a2-5ed6-4cf4-8e55-e45438eb39ba.png)

There are noticeably much less fraudulent job openings than non-fraudulent ones. 

From primary analysis, we know that the data have the following types;

    telecommuting: Boolean

    has_company_logo: Boolean

    has_questions: Boolean fraudulent: Boolean

    location: Categorical

    department: Categorical

    employment_type: Categorical
  
    required_experience: Categorical

    required_education: Categorical

    industry: Categorical

    function: Categorical

![% of fraudulent postings by telecommuting](https://user-images.githubusercontent.com/104303593/164986018-212b68eb-b6a9-4501-825b-c915b7b8e240.png)
Jobs that allow for telecommuting are twice as likely of being fraudulent.  

![% of fraudulent postings by company logo](https://user-images.githubusercontent.com/104303593/164986042-6ce3358b-3b33-4aaa-8dd8-51f0b3ab161c.png)
Job postings without a company logo are 8 times more likely of being a fraud.

![% of fraudulent postings by employment type](https://user-images.githubusercontent.com/104303593/164986066-e3490a66-3b56-4e7e-8d2d-1359aff66cc3.png)
Part-time employment postings have the highest occurrence of being fraudulent, followed second by other employment types and third by full-time employment.

![% of fraudulent postings by required education](https://user-images.githubusercontent.com/104303593/164986085-4315e721-4ea3-4d6e-be1c-92999fe1482a.png)
We can also see that almost 70% of postings that require some high school coursework is a fraud posting. Certification is also high, at about 10%. 

We drew a few conclusions from the EDA:

job_id can be removed because it serves as merely a label
There are a few variables ('department' and 'salary-range') with a noticeable amount of missing values, hence we decided to drop them.
We identified 'title', 'company_profile', 'description' and 'requirements' and 'benefits' as text data because they vary widely across different job openings.
Since department and salary_range have a lot of missing data (64.6% and 84.0% respectively), they are not reliable measures to train the model with. Hence we decided to drop them.

### Textual Analysis
![word cloud for fraudulent](https://user-images.githubusercontent.com/104303593/164986104-0e2bb4ad-bd73-43f5-a810-fbaa4c99f605.png)
![world cloud for not fraudulent](https://user-images.githubusercontent.com/104303593/164986119-7e7cca7d-2cfb-44e5-b925-90c59b505868.png)

For our textual data, we used word clouds to visualise the frequency of words appearing. For non-fraudulent postings, words like “work”, “customer” and “ experience” appeared most frequently. On the other hand, for fraudulent job postings, words like "customer service", "communication skills", "years experience" and "social media" appeared more frequently. This clear distinction in word usage between the 2 shows us that text is in fact an important factor for classification. 


## Cleaning and Pre-Processing
Cleaning of the textual data for the first model and cleaning of the categorical data for the second model. 

For cleaning of the boolean and categorical data, their data types had to be changed to the right ones and the undefined, NaN values had to be filled with blank spaces. 

For the cleaning of the textual data, the natural language processing pipeline is followed.
1. Retaining the textual data
2. Making all of the text lowercase
3. Tokenization - Tokenization is the process of separating text into smaller units, called tokens, and in this case the tokens are individual words.
4. Lemmatization - Lemmatization converts the words into their base form.
5. Stopword removal - Stop words, such as a, the, an, etc. that add little meaning to the text were removed.


## Textual Analysis Model
The textual classification involves 4 steps -  
1. Splitting the data into training and testing data - The train test split was done using a 70:30 ratio, where 30% of the data was for testing.

2. Vectorization using the TF-IDF Vectorizer - The text must be converted to a vector of numbers in order for the machine learning model to interpret it. The vectorizer used is Term Frequency-Inverse data frequency vectorizer which assigns values to each word based on a formula that measures the importance of the word in the text. 

3. Classification using the Support Vector Machine 

4. Analysis of the model using the metrics - This is a supervised machine learning model that uses classification algorithms for two-group classification problems. 
  The way it works:
  - Plot each data point in an n-dimensional space based on the number of features extracted. 
  - Value of each feature is the value of a particular coordinate. These come from the values generated by the vectorizer
  - Classify by finding the hyperplane that clearly differentiates the two classes 

![confusion matrix for nlp](https://user-images.githubusercontent.com/104303593/164986255-b6efba89-f70a-41af-8a92-5daf538f889a.png)

### Analysis 
The model is run on the test data to see how it performs based on various metrics. As seen in the confusion matrix, most of the non-fraudulent data was predicted as non-fraudulent. There are also 0 false positives. However, there are quite a few false negatives which are concerning as for this problem, it is better to classify a non-fraudulent job as fraudulent than classifying a fraudulent job as non-fraudulent.

## Categorical classification model

Firstly using get_dummies on pandas, we were able to manipulate and convert our categorical data into dummy variables based on the number of categories each variable had and we ended up with 204 dummy categories 

Next, we created  a forest_test function, using this we were able to separate our input data into training and test sets. We used a 30:70 ratio where we obtain 70% of the training sets from the main input and the remaining 30% made up the test sets.

We then planned to use a random forest classifier which grows multiple decision trees which are merged together for a more accurate prediction. The logic behind the Random Forest model is that multiple uncorrelated models (the individual decision trees) perform much better as a group than they do alone especially when there is low (or no) correlation between the individual decision trees which in our case are formed using our categorical data that are broken down into dummy categories. The predictions are either fraudulent or non-fraudulent and the random forest classifiers eventually takes the average of these predictions to make its prediction. 

![confusion matrix for random forest(full data)](https://user-images.githubusercontent.com/104303593/164986123-b779ba37-9ebe-4f70-bc6b-519182bf97c9.png)
![confusion matrix for random forest (first half of data)](https://user-images.githubusercontent.com/104303593/164986136-c6cba9b0-7c40-4036-9052-739436e4304b.png)
![confusion matrix for random forest (second half of data)](https://user-images.githubusercontent.com/104303593/164986146-6e05c1d4-8e85-4529-a5d2-5de754a1f185.png)

After training the random forest classifier using all the features, we tested it against the entire data set, the first half of the data set as well as the second half of the data set. The classifier proved to be consistent throughout, attaining a classification accuracy of 97-98%

Next, we adopted feature extraction using Principal Component Analysis which is a linear dimensionality reduction technique. It takes our original data and tries to find a combination of the input features which can best summarize the original data distribution so that its reduces its original dimensions. Eventually we are able to reduce our original data to just two variables

After running random forest on the data set created using PCA, we concluded that the addition of more features the model is better at predicting fraudulent jobs. When we ran the Random Forest Classifier using the dataset containing the 2 features constructed by PCA we achieved a 96% classification accuracy and upon using a dataset that we constructed adding another feature, we achieved a 97% classification accuracy

![confusion matrix for 2 features, pca data](https://user-images.githubusercontent.com/104303593/164986229-dce3be15-c678-45e5-9ae1-1ca3d7848379.png)
![confusion matrix for 3 features, pca data](https://user-images.githubusercontent.com/104303593/164986240-019280d1-6f3a-4333-9c38-455107545c40.png)


![2d scatter plot of the pca data](https://user-images.githubusercontent.com/104303593/164986215-8e5cd291-7e91-400f-8b29-0429dfd02a2b.png)
### Analysis of categorical classification using the random forest classifier
We noticed that there was noticeable overlapping of fraudulent and non-fraudulent jobs when we created a 2D scatterplot of the PCA data set’s distribution. This can be due to a few reasons: The first being that the big contrast between the large number of actual non-fraudulent data sets and fraudulent data sets make it such that the model is not trained to be as efficient as it can be.
The second being that a fraudulent categorical data can quite easily mirror a non-fraudulent categorical data, for instance “marketing” or “sales”

## Conclusion 
When we put the two models side by side, the support vector machine model excels in the way that it did not generate any false positive results while the random forest model was able to generate fewer false negative results. Both models were able to attain high classification accuracy of 96 and 97 percent respectively. However both models display low to moderate levels of recall value for “fraudulent” which is the ability of the classifier to find the fraudulent jobs. This could be due to the significantly low fraudulent jobs compared to non-fraudulent job openings which i previously mentioned and hence the classifiers cannot be trained optimally. A possible solution would be to combined both the random forest classifier together with the support vector classifier to patch up these flaws. If we had to give one model the edge, it would probably be the support vector machine classifier because it did not wrongly classify non-fraudulent jobs as fraudulent jobs which is what we were trying to achieve at the start of this project. Furthermore, text data is more organic and original and thus might serve as a clearer indicator of fraudulent and non-fraudulent jobs

## Takeaways 
1. Pandas Profiling
2. Natural Language Processing Pipeline
3. Vectorization: IF-IDF Vectorizer  
4. Support Vector Classifier
5. Random Forest Classifier 
6. Principal Component Analysis 
7. Feature Extraction

## Individual Contribution
Abhishekh Pandey - EDA and video recording and editing 

Anushree Arora - Textual Classification model and video recording 

Bryan Oh Wei Jie - Categorical Classification model and video recording 


## References
1. https://www.datacamp.com/community/tutorials/wordcloud-python 
2. https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76 
3. https://towardsdatascience.com/how-to-build-your-first-spam-classifier-in-10-steps-fdbf5b1b3870 
4. https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/ 
5. https://pythonspot.com/nltk-stop-words/ 
6. https://www.greycampus.com/opencampus/machine-learning/different-types-of-classifiers  
7. https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 
8. https://towardsdatascience.com/text-classification-in-python-dd95d264c802 
9. https://stackabuse.com/removing-stop-words-from-strings-in-python/ 
10. https://becominghuman.ai/spam-mail-detection-using-support-vector-machine-cdb57b0d62a8 
11. https://www.machinelearningplus.com/nlp/lemmatization-examples-python/ 
12. https://www.guru99.com/stemming-lemmatization-python-nltk.html 


