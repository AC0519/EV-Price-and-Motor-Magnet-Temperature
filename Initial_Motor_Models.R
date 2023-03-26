library(tidyverse)
library(AppliedPredictiveModeling)
library(caret)
library(e1071) 
library(corrplot)
library(lattice)
library(caTools)
library(factoextra)
library(glmnet)
library(MASS)

df <- motor_data
df$profile_id <- as.factor(df$profile_id)
#moves stator_winding to front of df for ease of comparison later
df <- dplyr::select(df, stator_winding, everything())


###PCA###
pc <- prcomp(df[-13],center = TRUE, scale = TRUE)

fviz_eig(pc)#scree plot of PCs
fviz_pca_var(pc)#vector plot of main contributors to pc1 and pc2


eig.val <- get_eigenvalue(pc) 
eig.val#percent of variance explained

pc_var <- get_pca_var(pc)
pc_var$contrib #contributions to the PCs



##########
#Linear Regression
#########

#Subset data into test and train
set.seed(42)
train <- sample(nrow(df), 0.7 * nrow(df))
training_data <- df[train,]
testing_data <- df[-train,]

#Simple linear regression attempting to predict stator_winding based on everything
Lm_everything <- lm(stator_winding ~ ., data = training_data)
summary(Lm_everything)
predicted_SW_everything <- predict(Lm_everything, newdata = testing_data)

testing_data$pred_SW_E <- predicted_SW_everything
testing_data <- dplyr::select(testing_data, stator_winding, pred_SW_E, everything())

rsme <- RMSE(testing_data$stator_winding, testing_data$pred_SW_E)
cat("RSME:", rsme, "\n")

ggplot(testing_data, aes(stator_winding, predicted_SW_everything))+
  geom_point()+
  stat_smooth(method=lm)

#with cv on lm
ctrl <- trainControl(method = "cv", number = 10)
LmCv <- train(stator_winding ~ ., data = training_data, method = 'lm', trControl = ctrl)
print(LmCv)
pred_cv <- predict(LmCv, newdata = testing_data)
testing_data$predCvSW <- pred_cv
testing_data <- dplyr::select(testing_data, stator_winding, predCvSW, everything())

rsme <- RMSE(testing_data$stator_winding, testing_data$predCvSW)
cat("RSME:", rsme, "\n")

#Linear regression attempting to predict Stator_Winding based on top variables from PCA
ctrl <- trainControl(method = "cv", number = 10)

#LM singular
Lm <- lm(stator_winding ~ stator_yoke+stator_tooth+coolant+pm+ambient, data = training_data)
#LM cross validated 
Lm <- train(stator_winding ~ stator_yoke+stator_tooth+coolant+pm+ambient, data = training_data, method = 'lm', trControl = ctrl)
Lm <- lm(stator_winding ~ stator_yoke+stator_tooth+coolant+pm+ambient, data = training_data)
summary(Lm)

predictedSW <- predict(Lm, testing_data)
testing_data$predictedSW <- predictedSW

rsme <- RMSE(testing_data$stator_winding, testing_data$predictedSW)
cat("RSME:", rsme, "\n")

plot(testing_data$stator_winding, testing_data$predictedSW)


ggplot(testing_data, aes(stator_winding, predictedSW))+
  geom_point()+
  stat_smooth(method=lm)



##########
#Elastic net Model
#########
#select only necessary values informed by Sid's unsupervised analysis
df <- motor_data %>% 
      dplyr::select('stator_tooth','stator_yoke','coolant','pm','ambient','stator_winding')

set.seed(42)
train <- sample(nrow(df), 0.7 * nrow(df))
training_data <- df[train,]
testing_data <- df[-train,]

fit <- cv.glmnet(x = as.matrix(training_data[, -5]), y = training_data$stator_winding, alph = 0.5)
predictions <- predict(fit, newx = as.matrix(testing_data[, -5]), s = "lambda.min")
#s = lambda.min selects the tuning parameter that gives min cross val error.  

print(fit)

pred <- as.data.frame(predictions)

pred <- pred %>% 
        dplyr::rename(predSW = lambda.min)

df <- cbind(testing_data, pred)

rsme <- RMSE(df$stator_winding, df$pred)
cat("RMSE:", rsme, "\n")

ggplot(df, aes(stator_winding, pred))+
  geom_point()+
  stat_smooth(method=lm)

# Varification of RMSE
actualResponse <- testing_data$stator_winding
rmse <- sqrt(mean((actualResponse - predictions)^2))
rmse

# Calculate R-squared
tss <- sum((actualResponse - mean(actualResponse))^2)
rss <- sum((actualResponse - predictions)^2)
r_squared <- 1 - (rss / tss)
r_squared


#####
#Robust linear regression
####

df <- motor_data %>% 
  dplyr::select('stator_tooth','stator_yoke','coolant','pm','stator_winding','ambient')


set.seed(42)
train <- sample(nrow(df), 0.7 * nrow(df))
training_data <- df[train,]
testing_data <- df[-train,]

formula <- stator_winding ~ stator_yoke+stator_tooth+coolant+pm+ambient

model <- rlm(formula, data = training_data, method = "MM")
summary(model)
cv <- trainControl(method = "cv", number = 10)
cv_model <- train(formula, data = training_data, method = "rlm", trControl = cv)
summary(cv_model)


predictedSW_RL <- predict(model, testing_data)
testing_data$predictedSW_RL <- predictedSW_RL

rsme <- RMSE(testing_data$stator_winding, testing_data$predictedSW_RL)
cat("RSME:", rsme, "\n")

predictedSW_RL <- predict(cv_model, testing_data)
testing_data$predictedSW_RL <- predictedSW_RL

rsme <- RMSE(testing_data$stator_winding, testing_data$predictedSW_RL)
cat("RSME:", rsme, "\n")


#The non-cross validated model returns a better initial RSME but the cross
#validated model performs better on the testing data.  





