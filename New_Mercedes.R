# New aproach for Mercedes Problem
set.seed(1)

require(ggplot2)
require(dplyr)
require(caret)
require(RANN)
require(mlr)
require(e1071)
require(rpart)
require(glmnet)
require(randomForest)
require(xgboost)
require(readr)
require(stringr)
require(car)
require(Matrix)
require(data.table)
dyn.load('/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home/jre/lib/server/libjvm.dylib')
require(rJava)
.jinit()
require(FSelector) # This is for variable selection based on entropy



require(parallel)
require(parallelMap)

train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)
sample_submission <- read.csv("sample_submission.csv", header = TRUE)

y_train <- train$y

x_train <- subset(train, select = -c(ID, y))
x_test <- subset(test, select = -c(ID))

len_train <- nrow(x_train)
len_test <- nrow(x_test)

train_test <- rbind(x_train, x_test)
# This command will take the transpose so the columns will be the rows
# and then return the rows (columns) that are not duplicated that is 
# why the ! sign. Basically we dropped the duplicated features
train_test <- train_test[, !duplicated(t(train_test))]

features <- colnames(train_test)

# This will encode the factor variables into numeric variables
for(f in features){
  if(class(train_test[[f]]) == "factor"){
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.numeric(factor(train_test[[f]], levels = levels))
  }
}

x_train <- train_test[1:len_train,]
x_test <- train_test[(len_train + 1):(len_train + len_test),]

dtrain <- xgb.DMatrix(as.matrix(x_train), label = y_train)
dtest <- xgb.DMatrix(as.matrix(x_test))

xgb_params <- list(colsample_bytree = 0.7,
                   subsample = 1,
                   eta = 0.005,
                   objective = 'reg:linear',
                   max_depth = 5,
                   num_parallel_tree = 1,
                   min_child_weight = 1,
                   base_score = mean(y_train))

best_nrounds <- 678

gbdt <- xgb.train(xgb_params, dtrain, best_nrounds)
prediction <- predict(gbdt, dtest)
sample_submission$y <- prediction
write.csv(sample_submission, "Prediction.csv", row.names = F)

            # Apparently this aproach was not the best it scored -0.00039
            # but there are some interesting facts about this code that can
            # be saved


# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 
# Now XGBoost with one hot encoding
df_train <- fread('train.csv', sep = ',', na.strings = 'NA')
df_test <- fread('test.csv', sep = ',', na.strings = 'NA')

data = rbind(df_train, df_test, fill = T) # Just put the train set over the test set

features <- colnames(data)

for(f in features){
  if( class(data[[f]]) == 'character' || (class(data[[f]]) == 'factor') ){
    levels = unique(data[[f]])
    data[[f]] = factor(data[[f]], levels = levels)
  }
}

# one-hot-encoding features
data = as.data.frame(data)
ohe_features <- c(names(data)[3:10])
dummies <- dummyVars(~ X0 + X1 + X2 + X3 + X4 + X5 + X6 + X8, data = data)
df_all_ohe <- as.data.frame(predict(dummies, newdata = data))
df_all_combined <- cbind(data[,-c(which(colnames(data) %in% ohe_features))],df_all_ohe)

data <- as.data.table(df_all_combined)

train <- data[data$ID %in% df_train$ID,] # Only select back the training set but already hot-encoded
y_train <- train[!is.na(y), y]
train <-  train[, y:=NULL]
train <- train[, ID:=NULL]
train_sparse <- data.matrix(train)

test <- data[data$ID %in% df_test$ID, ]
test_ids <- test[, ID]
test[, y:=NULL]
test[, ID:=NULL]
test_sparse <- data.matrix(test)

dtrain <- xgb.DMatrix(data = train_sparse, label = y_train)
dtest <- xgb.DMatrix(data = test_sparse)

gc()

# Parameters for XGBoost
param <- list(booster = 'gbtree',
              eval_metric = 'rmse',
              objective = 'reg:linear',
              eta = 0.1,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = 0.7)
              
rounds <-  52
mpreds <- data.table(id = test_ids)

for(random.seed.num in 1:10) {
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain,
                         params = param,
                         watchlist = list(train = dtrain),
                         nrounds = rounds,
                         verbose = 1,
                         print.every.n = 5)
  
  vpreds = predict(xgb_model,dtest) 
  mpreds = cbind(mpreds, vpreds)    
  colnames(mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

mpreds_2 <- mpreds[, id:=NULL]
mpreds_2 <- mpreds_2[, y:=rowMeans(.SD)]

submission <- data.table(ID=test_ids, y = mpreds_2$y)
write.table(submission, 'for_mercedes.csv', sep = ',', dec = ".", quote = FALSE, row.names = FALSE)

              # This submission scored 0.54695

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 
# Now try with the magic feature
trainp2 <- read_csv("train.csv")
testp2 <- read_csv("test.csv")

# We use the feature X0
# But first, get the average of "y" for each "X0"
meanX0 <- aggregate(trainp2$y, by = list(trainp2$X0), FUN = mean)
colnames(meanX0) <- c("X0", "meanX0")
trainmean <- trainp2[, c("ID", "y", "X0")]

trainF <- merge(trainmean, meanX0, by = 'X0')

# Some merge for the test
testmean <- testp2[, c("ID", "X0")]

testF <- merge(testmean, meanX0, by = "X0", all.x = T)

# Replace the NA's with the average:
testF[is.na(testF)] <- 100.97

# Submission
sub <- testF[, c("ID", "meanX0")]
sub <- sub[order(sub$ID), ]
colnames(sub) <- c("ID", "y")
write.csv(sub, "For.Mercedes.csv", row.names = F)

            # This obviously scored 0.54450

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 

# First substract the best features
# Begin with the encoded data set
train
test
y_train
dtrain   # Is the matrix version of train
dtest    # Matrix version of test
# This time test another model
parameters <- list(booster = 'gbtree',
                   eval_metric = 'rmse',
                   objective = 'reg:linear',
                   eta = 0.1,
                   max_depth = 10,
                   nround = 100,
                   subsample = 0.5,
                   colsample_bytree = 0.5,
                   nthread = 3,
                   max_depth = 6,
                   min_child_weight = 1,
                   alpha = 1)
                   
model_xgb <- xgb.train(data = dtrain,
                       params = parameters,
                       watchlist = list(train = dtrain))
                       
    # Somehow this crashed R itself

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 

train_2 <- data[data$ID %in% df_train$ID,]
# Creating tasks
trainTask <- makeRegrTask(data = train_2, target = 'y')

# We will select the best features like before
im_feat <- generateFilterValuesData(trainTask, method = c("information.gain"))
plotFilterValues(im_feat, n.show = 20)
important_f <- arrange(im_feat$data, desc(information.gain))
filtered.task <- filterFeatures(trainTask, method = 'information.gain', abs = 25)
new_train <- filtered.task$env$data
mejores_var <- names(new_train)[-26]
# Get the best features for the test set
new_test <- as.data.frame(data[data$ID %in% df_test$ID,])
new_test <- new_test[, mejores_var]

# Now we have the best features and also encoded within these data sets:
new_train
new_test
y_train 
train_sparse_2 <- data.matrix(new_train)
test_sparse_2 <- data.matrix(new_test)

# Now applying a simple XGBoost
dtrain_2 <- xgb.DMatrix(data = train_sparse_2, label = y_train)
dtest_2 <- xgb.DMatrix(data = test_sparse_2)

params <- list(booster = 'gbtree',
               eval_metric = 'rmse',
               objective = 'reg:linear',
               eta = 0.1,
               gamma = 1,
               max_depth = 4,
               min_child_weight = 1,
               subsample = 0.7,
               colsample_bytree = 0.7)

new_mpreds <- data.table(id = test_ids)

for(random.seed.num in 1:10){
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain_2,
                         params = params,
                         watchlist = list(train = dtrain_2),
                         nrounds = 52,
                         verbose = 1,
                         print.every.n = 5)
  vpreds = predict(xgb_model, dtest_2)
  new_mpreds = cbind(new_mpreds, vpreds)
  colnames(new_mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

new_mpreds_2 <- new_mpreds[, id:=NULL]
# Calculate th standard deviation for every row
new_mpreds_2 <- new_mpreds_2[, y:=rowMeans(.SD)]
submission <- data.table(ID = test_ids, y = new_mpreds_2$y)
write.table(submission, 'for_meche.csv', sep = ',', dec = ".", quote = FALSE, row.names = FALSE)

                # This submission scored 0.52632

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <-
# Lets try with the outliers (y < 120)
p <- new_train %>% filter(y < 120)
ggplot(data = p, aes(x = y)) + geom_histogram(binwidth = 1)
new_train <- p
new_y <- new_train$y
new_train$y <- NULL

train_sparse_2 <- data.matrix(new_train)
test_sparse_2 <- data.matrix(new_test)

# Now applying a simple XGBoost
dtrain_2 <- xgb.DMatrix(data = train_sparse_2, label = new_y)
dtest_2 <- xgb.DMatrix(data = test_sparse_2)

params <- list(booster = 'gbtree',
               eval_metric = 'rmse',
               objective = 'reg:linear',
               eta = 0.1,
               gamma = 1,
               max_depth = 4,
               min_child_weight = 1,
               subsample = 0.7,
               colsample_bytree = 0.7)

new_mpreds <- data.table(id = test_ids)

for(random.seed.num in 1:10){
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain_2,
                         params = params,
                         watchlist = list(train = dtrain_2),
                         nrounds = 52,
                         verbose = 1,
                         print.every.n = 5)
  vpreds = predict(xgb_model, dtest_2)
  new_mpreds = cbind(new_mpreds, vpreds)
  colnames(new_mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

new_mpreds_2 <- new_mpreds[, id:=NULL]
# Calculate th standard deviation for every row
new_mpreds_2 <- new_mpreds_2[, y:=rowMeans(.SD)]
submission <- data.table(ID = test_ids, y = new_mpreds_2$y)
write.table(submission, 'for_meche2.csv', sep = ',', dec = ".", quote = FALSE, row.names = FALSE)

                # This submission scored around 0.52...

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 

# The last good score was with ensemble algorithms, lets try this with the best features
train
test
test_ids
y_train

mejores_var[-c(25,24,23)] # These are the best parameters
best_var <- c(mejores_var[-c(25,24,23)], names(train)[369:421])
best_train <- as.data.frame(train)
best_train <- best_train[, best_var]

best_test <- as.data.frame(test)
best_test <- best_test[, best_var]

train_sparse_2 <- data.matrix(best_train)
test_sparse_2 <- data.matrix(best_test)

# Now applying a simple XGBoost
dtrain_2 <- xgb.DMatrix(data = train_sparse_2, label = y_train)
dtest_2 <- xgb.DMatrix(data = test_sparse_2)

params <- list(booster = 'gbtree',
               eval_metric = 'rmse',
               objective = 'reg:linear',
               eta = 0.1,
               gamma = 1,
               max_depth = 4,
               min_child_weight = 1,
               subsample = 0.7,
               colsample_bytree = 0.7)

new_mpreds <- data.table(id = test_ids)

for(random.seed.num in 1:10){
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain_2,
                         params = params,
                         watchlist = list(train = dtrain_2),
                         nrounds = 52,
                         verbose = 1,
                         print.every.n = 5)
  vpreds = predict(xgb_model, dtest_2)
  new_mpreds = cbind(new_mpreds, vpreds)
  colnames(new_mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

new_mpreds_2 <- new_mpreds[, id:=NULL]
# Calculate th standard deviation for every row
new_mpreds_2 <- new_mpreds_2[, y:=rowMeans(.SD)]
submission <- data.table(ID = test_ids, y = new_mpreds_2$y)
write.table(submission, 'for_meche3.csv', sep = ',', dec = ".", quote = FALSE, row.names = FALSE)

              # This algorithm scored 0.54209

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 

# Make some ensemble
fitControl <- trainControl(method = 'repeatedcv',
                           repeats = 5,
                           savePredictions = 'final')
fitControl2.1 <- trainControl(method = 'cv',
                              number = 10, 
                              savePredictions = 'final')


svm_model <- caret::train(best_train, y_train, method = 'svmRadial',
                          tuneLength = 9, metric = 'Rsquared', trControl = fitControl)

grid <- expand.grid(sigma = c(0.1, 0.1666, 0.2),
                    C = c(62, 63, 64, 65, 66))

svm_mdoel <- caret::train(best_train, y_train, method = 'svmRadial',
                          tuneGrid = grid, metric = 'Rsquared', trControl = fitControl)
model_svm_2 <- caret::train(best_train, y_train, method = 'svmRadial',
                            tuneGrid = grid, metric = 'Rsquared', trControl = fitControl2.1)



model_knn <- caret::train(best_train, y_train, method = 'knn',
                          tuneLength = 9, metric = 'Rsquared', trControl = fitControl)
model_knn_2 <- caret::train(best_train, y_train, method = 'knn',
                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl2.1)



model_glm <- caret::train(best_train, y_train, method = 'glm',
                          tuneLength = 9, metric = 'Rsquared', trControl = fitControl2.1)

# model_lasso <- caret::train(best_train, y_train, method = 'lasso',
#                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl2.1)

# model_ridge <- caret::train(best_train, y_train, method = 'ridge',
#                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl)

# model_bayes <- caret::train(best_train, y_train, method = 'brnn',
#                            tuneLength = 9, metric = 'Rsquared', trControl = fitControl)

best_train$pred_svm <- model_svm_2$pred$pred[order(model_svm_2$pred$rowIndex)] 
best_train$pred_knn <- model_knn_2$pred$pred[order(model_knn_2$pred$rowIndex)]
best_train$pred_glm <- model_glm$pred$pred[order(model_glm$pred$rowIndex)]


best_test$pred_svm <- predict(model_svm_2$finalModel, best_test[, best_var], type = 'response')
best_test$pred_knn <- predict(model_knn_2$finalModel, best_test[, best_var], type = 'response')
best_test$pred_glm <- predict(model_glm$finalModel, best_test[, best_var], type = 'response')





