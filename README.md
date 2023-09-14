# ArmMorph
logistic regression model using pandas and scikitlearn.

Small linguistics course project, purpose of this was to train a model to detect the verb form paradigms present in Eastern Armenian, for purposes of language learning. Idea was to examine morphological processes that may happen to the infinitive tense of the verb and examine the sorts of patterns that appear. Ideally, these patterns could help with recognizing and differentiating between tenses for a beginner learner, non-native speaker of Armenian.

A fully trained model that detects these verbs accurately would help in learning the Armenian language as it could both:

1) Educate on the morphological properties of the language overall
2) Showcase typical linguistic patterns
   
This is an elementary approach that likely needs heavy revision before it can be used practically.

# Data Sources
Universal Dependencies: https://github.com/UniversalDependencies/UD_Armenian-ArmTDP

Stop Words: https://github.com/stopwords-iso/stopwords-hy/blob/master/stopwords-hy.txt

# Results

![image](https://github.com/esafarian2/ArmMorph/assets/78068346/4bd58ba4-1856-4cd4-ad1e-30f43501cb53)


                precision   recall    f1-score  support
       Subcat      0.99      1.00      1.00      1342
       Tense       0.00      0.00      0.00         1
       Voice       0.00      0.00      0.00         6
       accuracy                        0.99      1349
       macro avg   0.33      0.33      0.33      1349
       weighted avg0.99      0.99      0.99      1349


# Conclusion:

The results from this project are definitely not conclusive. This is an extremely elementary application of this project. I hope to expand on this in the future but for now, it serves as my first introduction to model development. 

I also learned a lot about Armenian verbs along the way, perhaps moreso than what the model offers.
