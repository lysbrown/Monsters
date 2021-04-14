## Purpose of Project
The purpose of this project was to accurately classify monsters based on the specifications provided in a dataset. This project was in response to the kaggle competition, "Ghouls, Goblins, and Ghosts... Boo!" (link: https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo/overview). This kaggle competition provided both the test and training datasets used in analysis.

## File Functions
The ClassifyMonstersExplore.R file includes formatting the dataset to prepare it for a multi-step logistic regression. Within the file, you will see two separate approaches to logistic regression. The StackModel.R file includes manually constructing a stack model based on several other types of prediction applied to the dataset. 

## Methods for Cleaning Data and Feature Engineering
This data was remarkably clean. There was no missing data in the test data set. For the logistic regression, we did create two response variables, "isGhost" and "isGoblin", that were binary and with which we could perform a multi step logistic regression for classification. For the stacked model, we received the percent likelihood of being a ghost, golin, or ghoul from several different processes and substituted these to be our covariates.  

## Methods for Predictions
The first method we tried was a multi-step logistic regression. The second method included using XGBoost on a stacked model. 

