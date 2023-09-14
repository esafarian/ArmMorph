# ArmMorph
logistic regression model using pandas and scikitlearn.

Small linguistics course project, purpose of this was to train a model to detect the verb form paradigms present in Eastern Armenian, for purposes of language learning. 

# Data Sources:
Universal Dependencies: https://github.com/UniversalDependencies/UD_Armenian-ArmTDP

Stop Words: https://github.com/stopwords-iso/stopwords-hy/blob/master/stopwords-hy.txt

# Results

![image](https://github.com/esafarian2/ArmMorph/assets/78068346/4bd58ba4-1856-4cd4-ad1e-30f43501cb53)


    precision    recall  f1-score   support

       Subcat       0.99      1.00      1.00      1342
       Tense       0.00      0.00      0.00         1
       Voice       0.00      0.00      0.00         6

    accuracy                           0.99      1349
    macro avg       0.33      0.33      0.33      1349
    weighted avg       0.99      0.99      0.99      1349


# Conclusion:

The results from this project are definitely not conclusive. This is an extremely elementary application of this project. I hope to expand on this in the future but for now, it serves as my first introduction to model development. 

I also learned a lot about Armenian verbs along the way, perhaps moreso than what the model offers.
