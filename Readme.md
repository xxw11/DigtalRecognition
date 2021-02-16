# Readme

- sklearn工具包（svm和knn机器学习算法和PCA算法）
- pandas进行对csv数据的读取
- numpy矩阵运算
- imblearn对原有数据进行SMOTE（Synthetic Minority Oversampling Technique）处理，能提高一些准确度

## svm

第一次直接使用svm fit  train然后预测test 直接丢kaggle

得到0.95539

默认核函数 rbf 

C = 1.0

gamma 为1 / (n_features * X.var()) 



第二次使用了PCA对数据主成分进行分析，

取90%，识别率为0.49275

取85%，识别率为0.51217

负优化，kaggle上有人说使用pca取64个主成分是不错的选择但是经过尝试识别率0.49514  （可能需要对svm参数进行调整）



第三次以后放弃使用PCA,注意到给出的数据集0~9的数量不一致，且数量有一定差距，考虑使用SMOTE（Synthetic Minority Oversampling Technique）合成少数类过采样

结果为0.95585略有提升



第四次使用svm网格搜索参数并交叉验证得到了 SVC(C=10,tol=0.0001)拟合效果比较不错能达到0.96360



## knn

试着跑了一次knn

knn and smote KNeighborsClassifier(n_jobs=4, n_neighbors=10, weights='distance')

效果还可以0.92307



## cnn

还在做