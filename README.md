Please add subfolder to path first
# PCA and Linear Regression Example with Matlab
The data has 4869 features, the task is to come with the best model that can predict the result in test data. 

The first thing to do is to see which dimensions influence the outcome the most. It's natural to run a PCA to pick the dimensions with the first 3 larggest variance. 
```matlab
[coefs,scores,variances] = pca(X) 
```
The function pca in matlab does pca for you. 
It returns 3 values. The second one is the new dimensions and the third one is the variances with an ascending order.  

