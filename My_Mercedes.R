# For Mercedes-Benz
set.seed(1)

require(dplyr)
require(caret)
require(RANN)
require(mlr)
require(e1071)
require(rpart)
dyn.load('/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home/jre/lib/server/libjvm.dylib')
require(rJava)
.jinit()
.jcall("java/lang/System", "S", "getProperty", "java.runtime.version")
require(FSelector) # This is for variable selection based on entropy
require(woe)
require(glmnet)
require(randomForest)
require(xgboost)
require(readr)
require(stringr)
require(car)
require(parallel)
require(parallelMap)
#Dont use
require(InformationValue)

      
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

# There are no missing values

ext <- length(names(train))
c(330,297,293,290,289,268,235,233,107,93,11) # these only have '0' as the unique level

# Giving the fact that the predictors are only binary and we have no missing values
# we can proceed with variable selection process
# First we select the integer predictors
numeric_train <- train[, 11:ext]

# Then we implement PCA
PCA <- prcomp(numeric_train)
PCA$rotation[1:5, 1:5]

# Plot the resultant principal components
biplot(PCA, scale = 0)
      # Look for the extreme ends 
# The prcomp() function also provides the facility to provide the standard deviation for 
# each principal component
std_dev <- PCA$sdev

# Compute variance
pr_var <- std_dev^2

# Check variance of the first
head(pr_var)

      # We aim to find the components which explain the maximum variance
      # The higher the variance the higher the information contained in those components

# Proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]
      # So this are the percentages of the variance described from the first to the 
      # last component

# How many components should we select for modeling
# With the help of a Scree plot:
plot(prop_varex, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", 
     type = "b")
      # In this plot shows that nearly 20 components explain the majority of the
      # variance in the dataset
# Lets do a confirmation check by plotting a cumulative variance plot. 
# Cumulative Scree plot:
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
      # Here you can see that between 40 to 50 components describe the 90% of
      # the variance
      # PC1-PC40 or PC1-PC50



# ||| Predictive modeling with PCAs
# Add a training set with principal components
train.data <- data.frame(y = train$y, PCA$x)

# We are interested in the first 50 PCAs
train.data <- train.data[, 1:51]

# Run a decision tree
rpart.model <- rpart(y ~ ., data = train.data, method = "anova")
rpart.model

# Transform test into PCA
test.data <- predict(PCA, newdata = test)
test.data <- as.data.frame(test.data)

# Select the first 50 components
test.data <- test.data[, 1:50]

# Make prediction on test data
rpart.prediction <- predict(rpart.model, test.data)

samp <- read.csv("sample_submission.csv", header = TRUE)
samp$y <- rpart.prediction

write.csv(samp, "Predictions.csv", row.names = FALSE)






# ||| Other model |||
# Now with mlr
# Create the task
trainTask <- makeRegrTask(data = train, target = 'y')
testTask <- makeRegrTask(data = test, target = 'y')

# First we will select the best features
im_feat <- generateFilterValuesData(trainTask, method = c("information.gain", "chi.squared"))
plotFilterValues(im_feat, n.show = 20)
important_f <- arrange(im_feat$data, desc(information.gain))  
important_f[1:50, ]
# Select the most important features with this
filtered.task <- filterFeatures(trainTask, method = 'information.gain', abs = 50)
filtered.task$env$data # Has the 50 most important features
# Now with the percentage
im_feat2 <- generateFilterValuesData(trainTask, method = "information.gain")
filtered.task2 <- filterFeatures(trainTask, fval = im_feat2, perc = 0.25)
filtered.task2$env$data # Here is the dataset that contains the 25% of the most important features
# Now lets see if we can perform a variable selection with a wraper method
otherTrain <- read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
otherTest <- read.csv("test.csv", header = TRUE, stringsAsFactors = FALSE)
    # Now creating dummy variables, first without stringAsFactors = FALSE
dmy <- dummyVars("~.", data = train, fullRank = T)
dmy2 <- dummyVars("~.", data = test, fullRank = T)
test_transformed <- data.frame(predict(dmy2, newdata = test))
train_transformed <- data.frame(predict(dmy, newdata = train))
trainTask2 <- makeRegrTask(data = train_transformed, target = 'y')  

ctrl <- makeFeatSelControlSequential(method = "sfs", alpha = 0.02) # Specify the method
rdesc <- makeResampleDesc("CV", iters = 10)
sfeats <- selectFeatures(learner = "regr.lm", task = trainTask2, resampling = rdesc,
                         control = ctrl, show.info = FALSE)


# It seems we have to work with the selection of variables of "filtered.task$env$data"

# ||| Making the Lasso model
# Lasso Regression
lasso.learner <- makeLearner("regr.cvglmnet", predict.type = 'response') 

# Set 3 fold cross validation
set_cv <- makeResampleDesc("CV", iters = 3L)

lasso.model <- train(lasso.learner, trainTask)
getLearnerModel(lasso.model)  

# Make prdictions
lasso.p.model <- predict(lasso.model, newdata = train)
head(lasso.p.model$data$response)

# Make the submission 
submit <- data.frame("ID" = test$ID, "y" = lasso.p.model$data$response)
write.csv(submit, "Predictions.csv", row.names = FALSE)



# Making the lasso model again
# Important features
train_important <- filtered.task$env$data
n_vars <- names(train_important)
n_vars <- n_vars[n_vars != "y"]
test_important <- test[, n_vars]

lasso.learner # We already have this learner
set_cv <- makeResampleDesc("CV", iters = 3L) # and this configuration
trainTask <- makeRegrTask(data = train_important, target = 'y')
lasso.model <- train(lasso.learner, trainTask)
# Make predictions
lasso.p.model <- predict(lasso.model, newdata = test_important)
submit <- data.frame("ID" = test$ID, "y" = lasso.p.model$data$response)
write.csv(submit, "Predictions.csv", row.names = FALSE)
              
            # This was a good model, it scored 0.51190, the first place score is 0.57542


# Making the ridge model
train_important # The important features
test_important
ridge.learner <- makeLearner("regr.cvglmnet", predict.type = 'response', alpha = 0) 
set_cv # With three fold cross-validation
getParamSet("regr.cvglmnet")
ridge.model <- train(ridge.learner, trainTask)
# Make predictions
ridge.p.model <- predict(ridge.model, newdata = test_important)
submit <- data.frame("ID" = test$ID, "y" = ridge.p.model$data$response)
write.csv(submit, "Predictions.csv", row.names = FALSE)



# Making the SVM model
svm.learner <- makeLearner("regr.svm", predict.type = 'response')
svm.model <- train(svm.learner, trainTask)
svm.p.model <- predict(svm.model, newdata = test_important)
          # This does not work
# But lest do it without mlr
SVM_model <- svm(y~., data = train_important, cross = 10)
SVM_predict <- predict(SVM_model, test_important)
# Apparently there is an issue with the levels of the factor variables in the
# test_important dataframe, there are more levels in the train_important dataframe
# that need to be added to the levels of the test_important set
niveles <- levels(train_important[, 1])[c(2,3,4,38)]
levels(test_important[, 1]) <- c(levels(test_important[, 1]), niveles)
niveles <- levels(train_important[, 3])[c(2,16,24,33,36)]
levels(test_important[, 3]) <- c(levels(test_important[, 3]), niveles)
# Running the models again the same message appeared, we will try the next:
# Check if the train and test sets must have the same levels
# Check if the other columns also differ in their levels

# Lets balance the levels of the train and test set
which((levels(test_important[,1]) %in% levels(train_important[,1]) == FALSE))
lvl <- levels(test_important[, 1])[c(3,5,11,18,25,40)]
levels(train_important[, 1]) <- c(levels(train_important[, 1]), lvl)
          # Now the levels of the first factor variable in the test and train set
          # are balanced

lvl <-levels(test_important[, 3])[c(2,4,10,23,41,42)]
levels(train_important[, 3]) <- c(levels(train_important[, 3]), lvl)
          # All the levels have been balanced


# Now lets make the SVM algorith all over again
trainTask2 <- makeRegrTask(data = train_important, target = 'y')
svm.learner <- makeLearner("regr.svm", predict.type = 'response')
svm.model <- train(svm.learner, trainTask2)
svm.p.model <- predict(svm.model, newdata = test_important)

# NOW THIS FINALLY WORKED
SVM_model <- svm(y~., data = train_important, cross = 10)
SVM_predict <- predict(SVM_model, test_important)
# LETS MAKE THE PREDICTION 
submit <- data.frame("ID" = test$ID, "y" = SVM_predict)
write.csv(submit, "Predictions.csv", row.names = FALSE)

          # This seemed to be an improvement but still in the 2224 position
          # The previous score was 0.51190 and now is 0.51699



# Now lest make some Ensemble 
# But first lets remove the X0 - X8 features
train_2 <- train_important[, -c(1,2,3,4)]
test_2 <- test_important[, -c(1,2,3,4)]

fitControl <- trainControl(method = "cv",
                           number = 10,
                           savePredictions = "final", # To save out of fold predictions for best parameter combinations
                           classProbs = F # To save the class probabilities of the out of fold predictions
                           ) 
predictors <- names(train_2)[-47]
outcomeName <- 'y'

# Now lets train a Random Fores and test its accuracy with the test data
model_lr <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'glm', 
                  trControl = fitControl, tuneLength = 3)
# Make the predictions
model_lr_predict <- predict(object = model_lr, test_2)
# Submit it
submit <- data.frame("ID" = test$ID, "y" = model_lr_predict)
write.csv(submit, "Predictions.csv", row.names = FALSE)

# Lets see with another model
# Giving a try with SVM
fitControl_2 <- trainControl(method = 'repeatedcv',
                             repeats = 5,
                             savePredictions = "final")









# This is a SVM with a Radial Basis Function Kernel
svm.tune <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'svmRadial',
                         tuneLength = 9, metric = 'Rsquared', trControl = fitControl_2)
          # Here we can see that the best C is 4 and the value for sigma is 0.02872
          # With an Rsquared of 0.5678608
          # So lets tune the grid for this values
grid <- expand.grid(sigma = c(0.01, 0.0287, 0.2137),
                    C = c(3.75, 3.9, 4, 4.1, 4.25))
# Train and tune the SVM
svm.tune <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'svmRadial',
                          tuneGrid = grid, metric = 'Rsquared', trControl = fitControl_2)

          # The best parameters were sigma = 0.01 and C = 3.75 with an
          # Rsquared of 0.5733239

# Now with the linear model
svm.tune2 <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'svmLinear',
                          metric = 'Rsquared', trControl = fitControl_2)
# Comparing the SVM's
rValues <- resamples(list(svm=svm.tune,svm.tune2))
rValues$values
summary(rValues)
bwplot(rValues,metric="Rsquared",ylab =c("linear kernel", "radial kernel"))	

# Now lets make the predictions with the best parameters
svm.tune_prediction <- predict(object = svm.tune, test_2)
submit <- data.frame("ID" = test$ID, "y" = svm.tune_prediction)
write.csv(submit, "Predictions.csv", row.names = FALSE)
          
          # This model scored 0.51800 and got the place 2497 

# Let's see the performance of knn
model_knn <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'knn',
                          tuneLength = 9, metric = 'Rsquared', trControl = fitControl_2)
          # This model had an Rsquared of 0.5624203 with k = 19
# Now with glm, it has no tunning parameters
model_glm <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'glm',
                          tuneLength = 9, metric = 'Rsquared', trControl = fitControl2.1)
          # Rsquared of 0.5461841
# Now with lasso                          
model_relaxo <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'relaxo',
                             tuneLength = 9, metric = 'Rsquared', trControl = fitControl_2)

relaxo_predict <- predict(model_relaxo, test_2) # This seems to be very inaccurate
                                                # 'cause it has negative numbers



    # Other lasso
model_lasso <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'lasso',
                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl_2)
    # Retrain with the best parameter and fitControl2.1
grid.lasso <- expand.grid(fraction = 0.3)
model_lasso <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'lasso',
                            tuneGrid = grid.lasso, metric = 'Rsquared', trControl = fitControl2.1)




# Let's try with Ridge
model_ridge <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'ridge',
                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl_2)
          # Best Rsquared of 0.5529380 with lambda = 0.005179475   
     # Retrain with the best parameter and fitControl2.1  
grid.ridge <- expand.grid(lambda = 0.005179475)
model_ridge <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'ridge',
                            tuneGrid = grid.ridge, metric = 'Rsquared', trControl = fitControl2.1)



# Now lets try with bayesian
model_bayes <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'brnn', 
                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl_2)
          # Best Rsquared of 0.5709388 with 2 neurons
     # Retrain the model with the new fitControl2.1
grid.bayes <- expand.grid(neurons = 2)
model_bayes <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'brnn', 
                            tuneGrid = grid.bayes, metric = 'Rsquared', trControl = fitControl2.1)

# Now we got enough models lets try some ensemble 
# Train the top layer with the predictions of the other bottom layers
train_2$pred_svm <- model_svm_2$pred$pred[order(model_svm_2$pred$rowIndex)]
train_2$pred_knn <- model_knn_2$pred$pred[order(model_knn_2$pred$rowIndex)]
train_2$pred_glm <- model_glm$pred$pred[order(model_glm$pred$rowIndex)]
train_2$pred_ridge <- model_ridge$pred$pred[order(model_ridge$pred$rowIndex)]
train_2$pred_bayes <- model_bayes$pred$pred[order(model_bayes$pred$rowIndex)]
train_2$pred_lasso <- model_lasso$pred$pred[order(model_lasso$pred$rowIndex)]


svm_pred <- predict(model_svm_2$finalModel, test_2[, predictors], type = 'response')
svm_pred2 <- predict(model_svm_2$finalModel, test_2[, predictors])
svm_pred3 <- predict(model_svm_2, test_2[, predictors])


test_2$pred_svm <- predict(model_svm_2$finalModel, test_2[, predictors], type = 'response')
test_2$pred_knn <- predict(model_knn_2$finalModel, test_2[, predictors], type = 'response')
test_2$pred_glm <- predict(model_glm$finalModel, test_2[, predictors], type = 'response')
test_2$pred_ridge <- predict(model_ridge, test_2[, predictors])
test_2$pred_bayes <- predict(model_bayes$finalModel, test_2[, predictors], type = 'response')
test_2$pred_lasso <- predict(model_lasso, test_2[, predictors])

# Now train the top layers
predictors_top <- names(train_2)[48:53]

# GBM as the top layer model
model_gbm <- caret::train(train_2[, predictors_top], train_2[, outcomeName], method = 'gbm',
                          metric = 'Rsquared', trControl = fitControl2.1, tuneLength = 3)

# Finally making the predictions
test_2$gbm_stacked <- predict(model_gbm, test_2[, predictors_top])

# New test predictions
submit <- data.frame("ID" = test$ID, "y" = test_2$gbm_stacked)
write.csv(submit, "Predictions.csv", row.names = FALSE)
            
                # This algorithm scored 0.55060, the previous was 0.51800
                # and got the 1981 place out of 3032


# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 
# Train all the models again with the new fitControl_2 configuration

# Modelo prueba
prueba_modelo <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'glm',
                              tuneLength = 9, metric = 'Rsquared', trControl = fitControl_2)

prueba_fit <- trainControl(method = "cv",
                           number = 5,
                           savePredictions = 'final')
prueba_modelo2 <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'glm',
                               tuneLength = 9, metric = 'Rsquared', trControl = prueba_fit)
              # These two mdoels have the same y with different trControl sets

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 

index <- createDataPartition(train_2$y, p=0.75, list=FALSE)
trainSet <- train_2[ index,]
testSet <- train_2[-index,]

fitControl2.1 <- trainControl(method = 'cv',
                           number = 10,
                           savePredictions = 'final')

model_svm_2 <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'svmRadial',
                            tuneGrid = grid, metric = 'Rsquared', trControl = fitControl2.1)
model_knn_2 <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'knn',
                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl2.1)
grid.knn <- expand.grid(k = 11)
model_knn_2 <- caret::train(train_2[, predictors], train_2[, outcomeName], method = 'knn',
                            tuneGrid = grid.knn, metric = 'Rsquared', trControl = fitControl2.1)

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 

# Now its only matter of getting some ideas to implement
# Lest test with the neural network being the top layer

model_knn <- caret::train(train_2[, predictors_top], train_2[, outcomeName], method = 'brnn',
                          metric = 'Rsquared', trControl = fitControl2.1, tuneLength = 3)
model_brnn <- model_knn

test_2$brnn_stacked <- predict(model_brnn, test_2[, predictors_top])

# New test predictions
submit <- data.frame("ID" = test$ID, "y" = test_2$brnn_stacked)
write.csv(submit, "Predictions.csv", row.names = FALSE)

            # This algorithm scored ~ 0.53 which was not an improvement












# Now trying the XGBoost algorithm
train_2_numeric <- train_2
train_2_numeric[, 48:53] <- NULL
for(i in 1:(ncol(train_2_numeric) - 1)){
  train_2_numeric[, i] <- as.numeric(train_2_numeric[, i])
}
y <- train_2_numeric[, ncol(train_2_numeric)]
model_xgb <- xgboost(data = data.matrix(train_2_numeric[, predictors]),
                     label = as.matrix(y),
                     eta = 0.1,
                     max_depth = 10,
                     nround = 10,
                     subsample = 0.5,
                     colsample_bytree = 0.5,
                     seed = 1,
                     eval_metric = "rmse",
                     objective = "reg:linear",
                     nthread = 3)

# Just in case, converting the test set
test_2_numeric <- test_2
test_2_numeric[, 47:54] <- NULL
for(i in 1:ncol(test_2_numeric)){
  test_2_numeric[, i] <- as.numeric(test_2_numeric[, i])
}
y_pred <- predict(model_xgb, data.matrix(test_2_numeric))
submit <- data.frame("ID" = test$ID, "y" = y_pred)
write.csv(submit, "Predictions.csv", row.names = FALSE)

# Until now we have the numeric dataframes with the selected features
# Set the parameters for cv the xgboost
params <- list(booster = 'gbtree', objective = 'reg:linear', eta = 0.3, gamma = 0, 
               max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1)
        # Lets calculate the best nround for this model, this function also returns
        # CV error
labels <- train_2_numeric$y
dtrian <- xgb.DMatrix(model.matrix(~.+0, data=train_2_numeric[, -47]), label = labels)
dtest <- xgb.DMatrix(model.matrix(~.+0, data=test_2_numeric))

xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T,
                stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)
# Seting new parameters
model_xgb2 <- xgboost(data = data.matrix(train_2_numeric[, predictors]),
                      label = as.matrix(y),
                      booster = "gbtree",
                      nthread = 3,
                      # parameters for TreeBooster
                      nrounds = 100,
                      eta = 0.1,
                      #
                      #gamma = 5,
                      max_depth = 6,
                      min_child_weight = 1,
                      subsample = 0.8,
                      colsample_bytree = 0.8,
                      scale_pos_weigth = 1,
                      lambda = 0,
                      alpha = 1, # It also helps for variable selection
                      # parameters for Learning Task
                      objective = "reg:linear")

y_pred <- predict(model_xgb2, data.matrix(test_2_numeric))                      
submit <- data.frame("ID" = test$ID, "y" = y_pred)
write.csv(submit, "Predictions.csv", row.names = FALSE) 
              # This prediction scored 0.53

importance_matrix <- xgb.importance(model = model_xgb2)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)
  # The important variables are:
c(31,13,3,10,4,5)













# Olvidando todo lo anterior
bstDense <- xgboost(data = as.matrix(train_2_numeric[, -47]), label = train_2_numeric$y, 
                    max.depth = 6, eta = 0.1, nthread = 3, nround = 2, objective = "reg:linear")
#  This is another way to put it
dtrain <- xgb.DMatrix(data = as.matrix(train_2_numeric[, -47]), label = train_2_numeric$y)
bstdMatrix <- xgboost(data = dtrain, max.depth = 6, eta = 0.1, nthread = 2, nround = 2,
                      objective = "reg:linear")
# With verbose = 2, it prints information about the tree
bst <- xgboost(data = dtrain, max.depth = 6, eta = 0.1, nthread = 2, nround = 2,
               objective = "reg:linear", verbose = 2)
# This are the default parameters
params <- list(booster = "gbtree", objective = "reg:linear", eta = 0.3, gamma = 0, max.depth = 6,
               min_child_weight = 1, subsample = 1, colsample_bytree = 1)
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T,
                print_every_n = 10, early.stop.rounds = 20, maximize = F)
         # Best iteration: 16
    min(xgbcv$evaluation_log$test_rmse_mean)
         # 8.346065 
# Basic prediction
pred <- predict(bst, as.matrix(test_2_numeric))
# For parameter tunning we will use mlr, first we create tasks
train_task <- makeRegrTask(data = train_2_numeric, target = "y")
# Create learner
lrn <- makeLearner("regr.xgboost", predict.type = "response")
lrn$par.vals <- list(objective = "reg:linear", eval_metric = "rmse", nrounds = 100L, eta = 0.1)
# Set parameter space
params <- makeParamSet(makeDiscreteParam("booster", values = c("gbtree", "gblinear")), 
                       makeIntegerParam("max_depth", lower = 3L, upper = 10L),
                       makeNumericParam("min_child_weight", lower = 1L, upper = 10L), 
                       makeNumericParam("subsample", lower = 0.5, upper = 1), 
                       makeNumericParam("colsample_bytree", lower = 0.5, upper = 1))

# Set resampling strategy
rdesc <- mlr::makeResampleDesc("CV", iters = 5L)
# Now set the optimization strategy, use random search to find the best parameters and build 10 models
ctrl <- mlr::makeTuneControlRandom(maxit = 10L)
# Ensure a parallel computation 
parallelStartSocket(cpus = detectCores())
# Parameter tunning
mytune <- mlr::tuneParams(learner = lrn, task = train_task, resampling = rdesc,
                          par.set = params, control = ctrl, show.info = T)
# See the parameters
mytune
# Set the hyperparameters
lrn_tune <- setHyperPars(lrn, par.vals = mytune$x)
# Train model
xgmodel <- train(learner = lrn_tune, task = train_task)
# Predict model
xgpred <- predict(xgmodel, newdata = test_2_numeric)
# Submition
submit <- data.frame("ID" = test$ID, "y" = xgpred$data$response)
write.csv(submit, "Predictions.csv", row.names = FALSE)


                        

