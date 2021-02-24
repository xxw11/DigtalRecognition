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



## nn

激活函数 relu

共4层神经网络

神经元分别为512 256 128 64



损失函数使用交叉熵损失函数

使用随机梯度下降法

参数为lr = 0.01,momentum=0.5

batch_size = 100



经过255个epoch

loss降到0.000

测试集上识别率为0.96850



```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)
        
    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
```



## cnn_0

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size = 5)
        self.pooing = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)
        
    def forward(self,x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x

```

卷积-relu-池化-卷积-relu-池化-全连接

loss: 0.051
Accuracy on test set: 98 %

0.97814



loss: 0.012
Accuracy on test set: 98 %

0.98207



loss: 0.002
Accuracy on test set: 98 %

0.98225



## cnn_1

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size = 3)
        self.conv3 = torch.nn.Conv2d(20,30,kernel_size = 2)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(120,64)
        self.fc2 = torch.nn.Linear(64,32)
        self.fc3 = torch.nn.Linear(32,10)
        
    def forward(self,x):
        batch_size = x.size(0)
#         x为张量，张量.size 取出维度  取0  得到就是样本数量 n 1 28 28
        x = x.view(batch_size,1,28,28)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

卷积-池化-relu-卷积-池化-relu-卷积-池化-relu-全连接-全连接-全连接

loss: 0.005
Accuracy on test set: 98 %

0.98092



## LeNet

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,6,kernel_size = 5,padding = 2)
        self.conv2 = torch.nn.Conv2d(6,16,kernel_size = 5)
        self.fc1 = torch.nn.Linear(16*5*5,120)
        self.fc2 = torch.nn.Linear(120,84)
        self.fc3 = torch.nn.Linear(84,10)
        
    def forward(self,x):
        batch_size = x.size(0)
#         x为张量，张量.size 取出维度  取0  得到就是样本数量 n 1 28 28
        x = x.view(batch_size,1,28,28)
        x = F.max_pool2d( F.relu(self.conv1(x)) , 2)
        x = F.max_pool2d( F.relu(self.conv2(x)) , 2)
        
        x = x.view(batch_size,-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

卷积-relu-平均降采样-卷积-relu-平均降采样-全连接-全连接-全连接

loss: 0.001
Accuracy on test set: 98 %

0.98285