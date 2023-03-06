library(tidyverse)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) 
library(corrplot)
library(lattice)
library(factoextra)

df <- motor_cleaned_beck
df <- df[-1]
df$profile_id <- as.factor(df$profile_id)


###PCA###
pc <- prcomp(df[-13],center = TRUE, scale = TRUE)

fviz_eig(pc)#scree plot of PCs
fviz_pca_var(pc)#vector plot of main contributors to pc1 and pc2


eig.val <- get_eigenvalue(pc) 
eig.val#percent of variance explained

pc_var <- get_pca_var(pc)
pc_var$contrib #contributions to the PCs


#Subset data into test and train
set.seed(100)
sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
train  <- df[sample, ]
test   <- df[!sample, ]

#torque as a function of everything
lmTorqueAllPredictors <- lm(torque ~ . , data = train)
summary(lmTorqueAllPredictors)

#prediction power of torque as a function of everything
PredTorque = predict(lmTorqueAllPredictors, test)
#prediction results show over 99% accuracy
lmValues1 = data.frame(obs = test$torque, pred = PredTorque)
defaultSummary(lmValues1)





