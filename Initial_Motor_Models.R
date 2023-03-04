library(tidyverse)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) 
library(corrplot)
library(lattice)
library(factoextra)

df <- motor_cleaned_beck
df <- df[-1]

###PCA###
pc <- prcomp(df[-13],center = TRUE, scale = TRUE)

fviz_eig(pc)#scree plot of PCs

fviz_pca_biplot(pc, repel = TRUE, #Viz of PCA
                col.var = "red")


eig.val <- get_eigenvalue(pc) 
eig.val#percent of variance explained

pc_var <- get_pca_var(pc)
pc_var$contrib #contributions to the PCs


#Subset data into test and train


#PCR 
set.seed(100)
pcrTune <- train(x = solTrainXtrans, y = solTrainY, method = "pcr", tuneGrid = expand.grid(ncomp = 1:35), trControl = ctrl)
pcrTune 


