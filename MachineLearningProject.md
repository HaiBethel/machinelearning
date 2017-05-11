
#### Practical Machine Learning Project 

###### Name: Vu Thanh Hai 

#### Summary of approach
* Load the data set
* Get rid of the blank and NA columns.
* Check for any Near-Zero-Value variables.
* Get rid of the irrelevant columns (first 7 columns)
* Use cross-validation method to built a valid model:
    * 70% for model building (train data) 
    * the rest of 30% of the data for cross-validating (validation data)
* Apply 2 models: decision tree (rpart package) and random forest. 
* Check for the accuracy and choose the model with more accurate one.
* Apply the model to estimate classes of 20 observations


```R
url.train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url.test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```


```R
training <- read.csv(url(url.train), na.strings = c("NA", "", "#DIV0!"))
testing <- read.csv(url(url.test), na.strings = c("NA", "", "#DIV0!"))
```

* Let's get rid of all the columns with the NAs values. 


```R
training <- training[, colSums(is.na(training))==0]
testing <- testing[, colSums(is.na(testing))==0]
```

* Let check to see if the two data sets have the same variables.


```R
colncheck<- colnames(training) == colnames(testing)
```


```R
sum(colncheck)
```


59


* It seems that there are only 59 variables have the similar names. So we need to see which variables are not the same. 


```R
colnames(training)[(which(colnames(training) != colnames(testing)))]
```


'classe'



```R
colnames(testing)[(which(colnames(training) != colnames(testing)))]
```


'problem_id'


* The 60th column of the training set which is the `classe` is not the same with the 60th column of the testing set. 

* Let's check the near-zero-value variables and get rid of them as they are not going to contribute to the predictions. We can check by using the following codes. I am not printing them out but the result shows no "near zero value" variables in the data. 


```R
library(caret)
```


```R
NZVtrain <- nearZeroVar(training, saveMetrics = TRUE)
```


```R
NZVtrain
```


```R
colnames(training)
```


<ol class=list-inline>
	<li>'X'</li>
	<li>'user_name'</li>
	<li>'raw_timestamp_part_1'</li>
	<li>'raw_timestamp_part_2'</li>
	<li>'cvtd_timestamp'</li>
	<li>'new_window'</li>
	<li>'num_window'</li>
	<li>'roll_belt'</li>
	<li>'pitch_belt'</li>
	<li>'yaw_belt'</li>
	<li>'total_accel_belt'</li>
	<li>'gyros_belt_x'</li>
	<li>'gyros_belt_y'</li>
	<li>'gyros_belt_z'</li>
	<li>'accel_belt_x'</li>
	<li>'accel_belt_y'</li>
	<li>'accel_belt_z'</li>
	<li>'magnet_belt_x'</li>
	<li>'magnet_belt_y'</li>
	<li>'magnet_belt_z'</li>
	<li>'roll_arm'</li>
	<li>'pitch_arm'</li>
	<li>'yaw_arm'</li>
	<li>'total_accel_arm'</li>
	<li>'gyros_arm_x'</li>
	<li>'gyros_arm_y'</li>
	<li>'gyros_arm_z'</li>
	<li>'accel_arm_x'</li>
	<li>'accel_arm_y'</li>
	<li>'accel_arm_z'</li>
	<li>'magnet_arm_x'</li>
	<li>'magnet_arm_y'</li>
	<li>'magnet_arm_z'</li>
	<li>'roll_dumbbell'</li>
	<li>'pitch_dumbbell'</li>
	<li>'yaw_dumbbell'</li>
	<li>'total_accel_dumbbell'</li>
	<li>'gyros_dumbbell_x'</li>
	<li>'gyros_dumbbell_y'</li>
	<li>'gyros_dumbbell_z'</li>
	<li>'accel_dumbbell_x'</li>
	<li>'accel_dumbbell_y'</li>
	<li>'accel_dumbbell_z'</li>
	<li>'magnet_dumbbell_x'</li>
	<li>'magnet_dumbbell_y'</li>
	<li>'magnet_dumbbell_z'</li>
	<li>'roll_forearm'</li>
	<li>'pitch_forearm'</li>
	<li>'yaw_forearm'</li>
	<li>'total_accel_forearm'</li>
	<li>'gyros_forearm_x'</li>
	<li>'gyros_forearm_y'</li>
	<li>'gyros_forearm_z'</li>
	<li>'accel_forearm_x'</li>
	<li>'accel_forearm_y'</li>
	<li>'accel_forearm_z'</li>
	<li>'magnet_forearm_x'</li>
	<li>'magnet_forearm_y'</li>
	<li>'magnet_forearm_z'</li>
	<li>'classe'</li>
</ol>



* The first 7 variables seem not to be relevant since they are not activities. We should only look at the activities to predict `classe`.


```R
training <- training[, -c(1:7)]
```


```R
set.seed(21281)
```


```R
inTrain <- createDataPartition(training$classe, p =0.7, list = FALSE)
```


```R
train <- training[inTrain,]
validation <- training[-inTrain,]
```


```R
dim(train); dim(validation)
```


<ol class=list-inline>
	<li>13737</li>
	<li>53</li>
</ol>




<ol class=list-inline>
	<li>5885</li>
	<li>53</li>
</ol>




```R
summary(train$classe)
```


<dl class=dl-horizontal>
	<dt>A</dt>
		<dd>3906</dd>
	<dt>B</dt>
		<dd>2658</dd>
	<dt>C</dt>
		<dd>2396</dd>
	<dt>D</dt>
		<dd>2252</dd>
	<dt>E</dt>
		<dd>2525</dd>
</dl>



* Since we are doing a classification prediction, two methods might be useful "Decision Tree" and "Random Forest". 

#### Predicting with decision tree


```R
rpartModFit <- rpart(classe ~ ., data=train, method="class")
```


```R
rpartpred <- predict(rpartModFit, validation, type="class")
```


```R
confusionMatrix(rpartpred, validation$classe)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1530  190   21   65   11
             B   54  681  125   99  122
             C   38  147  771   75   75
             D   35   59   68  650   79
             E   17   62   41   75  795
    
    Overall Statistics
                                             
                   Accuracy : 0.7523         
                     95% CI : (0.741, 0.7632)
        No Information Rate : 0.2845         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.6856         
     Mcnemar's Test P-Value : < 2.2e-16      
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9140   0.5979   0.7515   0.6743   0.7348
    Specificity            0.9318   0.9157   0.9311   0.9510   0.9594
    Pos Pred Value         0.8420   0.6300   0.6971   0.7295   0.8030
    Neg Pred Value         0.9646   0.9047   0.9466   0.9371   0.9414
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2600   0.1157   0.1310   0.1105   0.1351
    Detection Prevalence   0.3088   0.1837   0.1879   0.1514   0.1682
    Balanced Accuracy      0.9229   0.7568   0.8413   0.8127   0.8471



```R
library(rattle)
```


```R
## We can see the decision tree visually but the print out is 
## quite crammed up. 
fancyRpartPlot(rpartFit, cex = 0.4)
```


![png](output_31_0.png)


* It seems that the decision might not be a good method; the accuracy is around 75%. Let's predict with the Random Forest. 


```R
library(randomForest)
rfModFit <- randomForest(classe~., data = train)
```


```R
## Getting the predicted value
rfpred <- predict(rfModFit, validation, type="class")
```


```R
confusionMatrix(rfpred, validation$classe)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1674    3    0    0    0
             B    0 1136    3    0    0
             C    0    0 1023    8    0
             D    0    0    0  954    1
             E    0    0    0    2 1081
    
    Overall Statistics
                                              
                   Accuracy : 0.9971          
                     95% CI : (0.9954, 0.9983)
        No Information Rate : 0.2845          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.9963          
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            1.0000   0.9974   0.9971   0.9896   0.9991
    Specificity            0.9993   0.9994   0.9984   0.9998   0.9996
    Pos Pred Value         0.9982   0.9974   0.9922   0.9990   0.9982
    Neg Pred Value         1.0000   0.9994   0.9994   0.9980   0.9998
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2845   0.1930   0.1738   0.1621   0.1837
    Detection Prevalence   0.2850   0.1935   0.1752   0.1623   0.1840
    Balanced Accuracy      0.9996   0.9984   0.9977   0.9947   0.9993


**Conclusion:** As we can see, the Random Forest method give us much better accuracy which is consistent with the theory which we learnt in the course. Therefore, we should use this model to classify the activities. 

* Now, let's apply the model `rfModfit` to the twenty observation in `testing` data. 


```R
FinalPred <- predict(rfModFit, testing, type="class")
```

**Reference:** Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. *Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)* . Stuttgart, Germany: ACM SIGCHI, 2013.
