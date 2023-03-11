library(tidyverse)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) 
library(corrplot)
library(lattice)
library(caTools)
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
sample <- sample.split(df$u_q, SplitRatio = .7)
dfTrain <- subset(df, sample==T)
dfTest <- subset(df, sample==F)

#PCR Regression
PCR <- train(x = dfTrain , y = endpointsTrain$V2, method = "pcr", 
             trControl = ctrl, tuneLength = 3) 




