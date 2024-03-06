library(BGLR)
library(caret)
library(ggplot2)

#data(wheat)
#X <- wheat.X
#Y <- wheat.Y
#A <- wheat.A

#write.csv(A, "C:/MULTI_TASK_LASSO/R code/A.csv")

#dimensions_X <- dim(X)
#dimensions_X
#dimensions_Y <- dim(Y)
#dimensions_Y
#dimensions_A <- dim(A)
#dimensions_A


set.seed(123)



mice <-read.csv('')

preproc <- preProcess(mice, method = c("center", "scale"))
preproc <- preProcess(mice, method = "range", rangeBounds = c(0, 1))
mice <- predict(preproc, mice)

X <- mice[, -(1:4)]
Y <- mice[, 1:4]
X <- as.matrix(X)
Y <- as.matrix(Y)



folds <- sample(rep(1:5, length.out=1813))


MSEs <- numeric(5)
MSEs1 <- numeric(5)
  

timing <- system.time({
  
  
  for (k in 1:5) {
    # k=2
    print(k)
    testIdx <- which(folds == k)
    #trainIdx <- setdiff(1:1813, testIdx)
    
    Ytest = Y[testIdx,]
    #Xtest = X[testIdx,] #X is always the same
    Ytrain = Y
    Ytrain[testIdx,] = NA #Impute the test y
    
    ETA2 <-list(list(X=X,model="SpikeSlab"))
    fm2 <- Multitrait(y=Ytrain,ETA=ETA2,nIter=6000,burnIn=1000)
    
    
    ypred<- fm2$ETAHat  
    
    Ypredtest<-ypred[testIdx]
    
    
    sqresid<-(Ytest-Ypredtest)^2
    
    
    MSEs[k] <- mean(sqresid)
    
  }
  
})
print(summary(fm2$fit))

print(timing)

mean(MSEs)  



cat("CPU time：", sum(cpu_time), "s\n")
current_time <- Sys.time()
formatted_time <- format(current_time, format = "%Y-%m-%d %H:%M:%S")
cat("current time：", formatted_time, "\n")