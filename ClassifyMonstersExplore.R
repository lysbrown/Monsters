##
## Classifying Monsters
##

###############
## Libraries ##
###############

library(aod)
library(tidyverse)

##################
## Read in Data ##
##################

train <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/train.csv")
test <- read.csv("/Users/LysLysenko/Desktop/Kaggle/Monsters/test.csv")
full <- bind_rows(train = train,test = test, .id = 'Set')

##############
## Cleaning ##
##############

full$color <- full$color %>% as.factor()
full$type <- full$type %>% as.factor()
unique(full$type)

#############################
## First Response Variable ##
#############################

full$isGhost <- numeric(length(full$color))

for(i in 1:length(full$color)) {
  if(full[i,8] %in% c('Ghost')){
    full$isGhost[i] = 1
  }
  else{
    full$isGhost[i] = 0 
  }
}

##############################
## Second Response Variable ##
##############################

full$isGoblin <- numeric(length(full$color))

for(i in 1:length(full$color)) {
  if(full[i,8] %in% c('Goblin')){
    full$isGoblin[i] = 1
  }
  else{
    full$isGoblin[i] = 0 
  }
}

####################
## Is Ghost Model ##
####################

ghostlogit <- glm(isGhost ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = full %>% filter(Set == 'train') %>% select(-Set, -isGoblin), family = binomial(link = "logit"))
ghostodds <- predict(ghostlogit, newdata = full %>% filter(Set == 'test'), type = 'response')


#####################
## Is Goblin Model ##
#####################

subfull <- full %>% filter(type %in% c('Goblin','Ghoul'))
goblinlogit <- glm(isGoblin ~ bone_length + rotting_flesh + hair_length + has_soul + color, data = subfull %>% select(-Set, -isGhost), family = binomial(link = "logit"))
goblinodds <- predict(goblinlogit, newdata = full %>% filter(Set == 'test'), type = 'response')

################################
## Conditional Logistic Model ##
################################

newtest <- full %>% filter(Set == 'test') %>% select(-isGhost, -isGoblin)
newtest <- cbind(newtest,ghostodds,goblinodds)

for(i in 1:nrow(newtest)){
  if(newtest[i,9] > 0.5){
    newtest[i,8] = "Ghost"
  }
  else if (newtest[i,10] > 0.5){
    newtest[i,8] = "Goblin"
  }
  else{
    newtest[i,8] = "Ghoul"
  }
}

submission <- newtest %>% select(id, type)
write_csv(submission, "/Users/LysLysenko/Desktop/Kaggle/Monsters/submission1.csv")

##########################
## Dr. Heaton's Version ##
##########################

# Here is my version.  I got a 72% accuracy

## Libraries
library(tidyverse)
library(vroom)
library(caret)
library(DataExplorer)

## Read in the data
train <- vroom("../Data/train.csv")
test <- vroom("../Data/test.csv")
ghost <- bind_rows(train,test)

## Set as factors
ghost$type <- as.factor(ghost$type)
ghost$color <- as.factor(ghost$color)

## Set up indicators
ghost <- ghost %>%
  mutate(isGhost=ifelse(type=="Ghost", "Yes", "No") %>% as.factor(),
         isGhoul=ifelse(type=="Ghoul", "Yes", "No") %>% as.factor())

## First Layer Logistic Regression (Ghost vs. Not Ghost)
ghost.logreg <- glm(isGhost~bone_length+rotting_flesh+hair_length+
                      has_soul+color,
                    data=ghost,
                    family=binomial)

## Second Layer Logistic Regression (Ghoul vs. Goblin)
ghoul.logreg <- glm(isGhoul~bone_length+rotting_flesh+hair_length+
                      has_soul+color,
                    data=ghost %>% filter(isGhost=="No"),
                    family=binomial)

## Get Predicted Probabilities for all data
predProbs <- data.frame(id=ghost$id,
                        ghostProb_LR=predict(ghost.logreg,
                                             newdata=ghost,
                                             type="response"),
                        ghoulProb_LR=(1-predict(ghost.logreg,
                                                newdata=ghost,
                                                type="response"))*
                          predict(ghoul.logreg, newdata=ghost,
                                  type="response"))
predProbs <- predProbs %>%
  mutate(goblinProb_LR=1-ghostProb_LR-ghoulProb_LR)

## Get Predicted Classes
predClasses <- data.frame(id=predProbs$id,
                          type=apply(predProbs[,-1], 1, function(x){
                            c("Ghost","Ghoul","Goblin")[which.max(x)]
                          }))
kaggleSubmission <- predClasses[-(1:nrow(train)),]
write.csv(x=kaggleSubmission, file="../Data/kaggleSubmission.csv",
          row.names=FALSE)


