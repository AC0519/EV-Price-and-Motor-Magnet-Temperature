library(tidyverse)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) 
library(corrplot)
library(lattice)
library(caTools)
library(factoextra)

df <- EV

df$Brand <- as.factor(df$Brand)
df$FastCharge_KmH <- as.double(df$FastCharge_KmH) #NAs introduced
df$RapidCharge <- as.factor(df$RapidCharge)
df$PowerTrain <- as.factor(df$PowerTrain)
df$PlugType <- as.factor(df$PlugType)
df$BodyStyle <- as.factor(df$BodyStyle)
df$Segment <- as.factor(df$Segment)

#how many NAs introduced by changing FastCharge_KmH to double
find_na <- is.na(df$FastCharge_KmH)
sum(find_na == TRUE)

# Five NAs were introduced. The EV guys can research this since we have the 
#majority of the data to probably find the answer or we can eliminate these rows.
#Elimination is probably a bad idea since we have minimal data to begin with