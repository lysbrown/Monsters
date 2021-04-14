##
## Building a Stack Model
##

## Libraries
library(caret)
library(tidyverse)


## Read in Data
train <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/train.csv")
test <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/test.csv")
full <- bind_rows(train = train,test = test, .id = 'Set')

logreg <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/LogRegPreds.csv")
nn <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/nn_Probs_65acc.csv")
mlp <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/multilayerperceptron.csv")
cla <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/classification_submission_rf.csv") %>%
  mutate(id = ID) %>% select(-ID)
gbm <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/probs_gbm.csv") %>%
  mutate(id = ID) %>% select(-ID)
knn <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/Probs_KNN.csv") %>%
  mutate(id = ID) %>% select(-ID)
svm <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/probs_svm.csv") %>%
  mutate(id = ID) %>% select(-ID)
xgb <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/xgbTree_probs.csv") %>%
  mutate(id = ID) %>% select(-ID)

all <- left_join(full, logreg, by = "id") %>%
  left_join(., nn, by = 'id') %>%
  left_join(., mlp, by = 'id') %>%
  left_join(., cla, by = 'id') %>%
  left_join(., gbm, by = 'id') %>%
  left_join(., knn, by = 'id') %>%
  left_join(., svm, by = 'id') %>%
  left_join(., xgb, by = 'id') 

all$type <- as.factor(all$type)
all$id <- as.factor(all$id)
  
## Principle Component Analysis
pp <- preProcess(all, method = 'pca')
all_pp <- predict(pp, all)

###########################
## XGBoost Stacked Model ##
###########################

#DataExplorer::plot_missing(all_pp)

## Grid Defaults
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

## Tuned Model

nrounds <- 1000

tune_grid <- expand.grid(
  nrounds = 350,
  eta = 0.3,
  max_depth = 6,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

## Train Control
train_control <- caret::trainControl(
  method = "cv",
  number=100,
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

## XGBoost Creation
xgb_tune <- caret::train(form=as.factor(type) ~ .,
                         data=all_pp %>%
                           filter(Set == 'train') %>% 
                           select(-id, -Set, -color),
                         method="xgbTree",
                         trControl = train_control,
                         tuneGrid = tune_grid,
                         verbose = TRUE
)
xgb_tune$bestTune

tune_preds <- predict(xgb_tune, all_pp %>% filter(Set == 'test') %>% select(-id, -Set, -color))
submission <- cbind(all_pp %>% filter(Set == 'test') %>% select(id), type = tune_preds) %>% as.data.frame()

write_csv(submission, "/Users/LysLysenko/Desktop/Kaggle/Monsters/stacksub2.csv")










