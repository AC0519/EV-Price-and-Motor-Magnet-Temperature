library(tidyverse)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) 
library(corrplot)
library(lattice)

df <- motor_cleaned_beck
df <- df[-1]

