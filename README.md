# apl405-project
SVM for artificial muscle use case classification

[For APL405{Machine Learning in Mechanics} course we took in Sem 2 - Spring 2022]

Usage:
each folder contains its own dataset and code file (.ipynb)

1. bivariate_svm: code for the 10 bivariate svm, each trained by taking 2 features at a time.
2. nn_raw: neural network code trained on raw input data, with NaN filled with Zero. [current accuracy: 32%]
3. nn_imputed: neural network code trained on imputed data. [current accuracy: 59%]
4. nn_augmented: neural network code trained on augmented data. [current accuracy: 75%]

The following are the dataset files:
nn_raw/data/original_data.csv: original data provided
nn_imputed/data/new_data.csv: imputed data, made by using impute library of SciKit Learn
nn_augmented/data/augmented_data.csv: augmented data, NaN filled with actual data from sources mentioned in report

Access:
final_report.pdf and final_presentation.pdf for better understanding
