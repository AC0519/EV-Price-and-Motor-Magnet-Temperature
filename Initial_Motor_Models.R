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

rmse <- RMSE(testing_data$stator_winding, testing_data$pred_SW_E)
cat("rmse:", rmse, "\n")

ggplot(testing_data, aes(stator_winding, predicted_SW_everything))+
  geom_point()+
  stat_smooth(method=lm)

plot(Lm_everything$residuals)

#with cv on lm
ctrl <- trainControl(method = "cv", number = 10)
LmCv <- train(stator_winding ~ ., data = training_data, method = 'lm', trControl = ctrl)
print(LmCv)
pred_cv <- predict(LmCv, newdata = testing_data)
testing_data$predCvSW <- pred_cv
testing_data <- dplyr::select(testing_data, stator_winding, predCvSW, everything())

rmse <- RMSE(testing_data$stator_winding, testing_data$predCvSW)
cat("rmse:", rmse, "\n")

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

rmse <- RMSE(testing_data$stator_winding, testing_data$predictedSW)
cat("rmse:", rmse, "\n")

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

rmse <- RMSE(df$stator_winding, df$pred)
cat("RMSE:", rmse, "\n")

ggplot(df, aes(stator_winding, pred))+
  geom_point()+
  stat_smooth(method=lm)

resid <- (df$stator_winding - df$predSW)
plot(resid)

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

rmse <- RMSE(testing_data$stator_winding, testing_data$predictedSW_RL)
cat("rmse:", rmse, "\n")

predictedSW_RL <- predict(cv_model, testing_data)
testing_data$predictedSW_RL <- predictedSW_RL

rmse <- RMSE(testing_data$stator_winding, testing_data$predictedSW_RL)
cat("rmse:", rmse, "\n")

#The non-cross validated model returns a better initial rmse but the cross
#validated model performs better on the testing data.  

# Varification of RMSE
actualResponse <- testing_data$stator_winding
rmse <- sqrt(mean((actualResponse - predictedSW_RL)^2))
rmse

# Calculate R-squared
tss <- sum((actualResponse - mean(actualResponse))^2)
rss <- sum((actualResponse - predictedSW_RL)^2)
r_squared <- 1 - (rss / tss)
r_squared

ggplot(testing_data, aes(stator_winding, predictedSW_RL))+
  geom_point()+
  stat_smooth(method=lm)

plot(resid(cv_model))


#########
#Logistic Regression to predict pass/fail
#########


#Assuming a NEMA class B insulation is the wire used in this motors data
#According to NEMA's website Class B is the most common class used in 60 cycle US motors

#Create a column of pass/fail based on a max temp allowance of the stator winding at 130 Celsius
df <- motor_data
df <- dplyr::select(df, -profile_id)
df$insulation_result <- ifelse(df$stator_winding > 130, "Failure", "Pass")

df %>% 
  group_by(insulation_result) %>% 
  summarize(total = n())

df$insulation_result <- as.factor(df$insulation_result)


set.seed(42)
train <- createDataPartition(df$stator_winding, p = 0.7, list = F)
dftrain <- df[train,]
dftest <- df[-train,]

dftrain %>% 
  group_by(insulation_result == "Failure") %>% 
  summarize(total = n())

dftest %>% 
  group_by(insulation_result == "Failure") %>% 
  summarize(total = n())

Trainx <- dftrain[,-13]
Trainy <- dftrain$insulation_result
Testx <- dftest[,-13]
Testy <- dftest$insulation_result

ctrl <- trainControl(method = "CV", number = 10)


logisticTune <- train(x = Trainx, y = Trainy, 
                      method = "multinom", metric = "Accuracy", 
                      trControl = ctrl)

logisticTune 

testResults <- data.frame(obs = Testy,
                          logistic = predict(logisticTune, Testx))

confusionMatrix(data = predict(logisticTune, Testx), 
                reference = Testy)

#initially tried this model on all data and the results are terrible.  Not a single failure was accurately #predicted by the model I am now going to try the same with the preselected data from the PCA analysis

df <- motor_data %>% 
  dplyr::select('stator_tooth','stator_yoke','coolant','pm','ambient','stator_winding')

df$insulation_result <- ifelse(df$stator_winding > 130, "Failure", "Pass")

df %>% 
  group_by(insulation_result) %>% 
  summarize(total = n())
#We had 2845 failures

df$insulation_result <- as.factor(df$insulation_result)


set.seed(42)
train <- createDataPartition(df$stator_winding, p = 0.7, list = F)
dftrain <- df[train,]
dftest <- df[-train,]

dftrain %>% 
  group_by(insulation_result == "Failure") %>% 
  summarize(total = n())

dftest %>% 
  group_by(insulation_result == "Failure") %>% 
  summarize(total = n())

Trainx <- dftrain[,-7]
Trainy <- dftrain$insulation_result
Testx <- dftest[,-7]
Testy <- dftest$insulation_result

ctrl <- trainControl(method = "CV", number = 10)


logisticTune <- train(x = Trainx, y = Trainy, 
                      method = "glm", metric = "Accuracy", 
                      trControl = ctrl)

logisticTune 

testResults <- data.frame(obs = Testy,
                          logistic = predict(logisticTune, Testx))

confusionMatrix(data = predict(logisticTune, Testx), 
                reference = Testy)


# Initial results were still atrocious.  However, switching to a glm as the training method instead of multinom gave 100% accurate results on the test data 

#Since this was directly based on temp, can I remove this variable and predict based on the others?
df <- motor_data
df$profile_id <- as.factor(df$profile_id)
df <- dplyr::select(df, stator_winding, everything())
df <- dplyr::select(df, -profile_id)
df$insulation_result <- ifelse(df$stator_winding > 130, "Failure", "Pass")
df$insulation_result <- as.factor(df$insulation_result)
df <- df %>% 
  dplyr::select('stator_tooth','stator_yoke','coolant','pm','ambient','insulation_result')



set.seed(42)
train <- createDataPartition(df$stator_tooth, p = 0.7, list = F)
dftrain <- df[train,]
dftest <- df[-train,]

dftrain %>% 
  group_by(insulation_result == "Failure") %>% 
  summarize(total = n())

dftest %>% 
  group_by(insulation_result == "Failure") %>% 
  summarize(total = n())

Trainx <- dftrain[,-6]
Trainy <- dftrain$insulation_result
Testx <- dftest[,-6]
Testy <- dftest$insulation_result

ctrl <- trainControl(method = "CV", number = 10)


logisticTune <- train(x = Trainx, y = Trainy, 
                      method = "glm", metric = "Accuracy", 
                      trControl = ctrl)

logisticTune 

testResults <- data.frame(obs = Testy,
                          logistic = predict(logisticTune, Testx))

confusionMatrix(data = predict(logisticTune, Testx), 
                reference = Testy)

