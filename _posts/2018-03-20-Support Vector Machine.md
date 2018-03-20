---
layout:     post
title:      SVM
subtitle:   Support Vector Machine
date:       2018-03-20
author:     Jiayi
header-img: img/post-bg-ios10.jpg
catalog: true
tags:
    - Python
    - SVM
---


## 1. SVM支持向量机：
The princile of SVM is to find out hyper plan between two classes of datasets.
![png](/img/svm/SVM01.png)

What this line does that the other ones don't do? It maximizes the distance to the nearest point, and it does this relative to both classes.

It's a line that maximizes the distance to the nearest points in either class, that distance is often called **margin**. The margin is the distance between the line and the nearest point of either of the two classes.
![png](/img/svm/SVM02.png)

Two key points:
1. SVM always consider whether the classification is correct or not, rather than maximizing the distance between datasets.
2. SVM maximizes the robustness of the classification.
3. SVM looks for the decision boundry that maxmizes the distance of two datasets, meanwhile tolerates specific outliner by parameter tuning.
![png](/img/svm/SVM03.png)
![png](/img/svm/SVM04.png)


## 2. SVM in SKLEARN

http://scikit-learn.org/stable/modules/svm.html


```python
from sklearn import svm
X = [[0, 0], [1, 1]]  # training feature
y = [0, 1]            # training label
clf = svm.SVC()       # create classifier
clf.fit(X, y)         # training

```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
clf.predict([[2., 2.]])   #the model can then be used to predict new values
```




    array([1])



### Coding up with SVM


```python
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

clf.fit(features_train, labels_train)

#### store your predictions in a list named pred
pred = clf.predict(features_test)    #only input features_test because the label is what we are trying to predict

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
```


```python
acc
```




    0.92000000000000004




```python
pred
```




    array([ 0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,
            0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,
            1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,
            0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,
            0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,
            1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,
            1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,
            1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,
            0.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,
            0.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,
            1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,
            0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,
            1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,
            0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,
            0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,
            1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
            1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,
            0.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,
            1.,  1.,  1.])



## 3. Non-Linear SVM

Z代表到原点的距离。将数据映射到带有Z的新坐标，会发现蓝圈Z值都大，红圈Z值都小。即新坐标中linearly separable

我们增加了一个特性Z，就能够让SVM学习圆形的非线性决策面

![](SVM05.png)

以下的图形，新增非线性特性|X|可以把红蓝圆圈线性分开
![](SVM06.png)

### kernal trick
但为了避免写一大推新的特征，我们需要使用**kernal trick**: 接受低维度输入空间或特征空间，将其映射到高维度空间。

应用核函数将输入空间从xy变换到更大的输入空间后，再使用SVM对数据点分类，得到解后返回原始空间，得到一个非线性分割。

![](SVM07.png)

## 4. SVM in SKLEARN
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

Parameter:
- Kernal
- Gamma, 单个训练样本，对结果作用范围的远近(define how far the influence of a single training example reaches)
    - 较小值意味着每个点都可能对最终结果产生作用
    - 较大值意味着训练样本对距离较近的决策边界有影响
- C, 在光滑的决策边界，以及尽可能正确分类所有训练点两者之间进行平衡. C值越大，就得到更复杂的决策边界值.

https://classroom.udacity.com/courses/ud120/lessons/2252188570/concepts/33843786680923


```python
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=1000)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

clf.fit(features_train, labels_train)

#### store your predictions in a list named pred
pred = clf.predict(features_test)    #only input features_test because the label is what we are trying to predict

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
```


```python
acc
```




    0.92400000000000004



## 5. 优缺点
优点： 
- 适用于有明显分隔边界的复杂数据

缺点： 
- 不适用于海量数据。训练时间与数据量的三次方成正比。
- 不使用于数据集噪声过大的情况（类之间的重叠多，且需要将不同类分开，naive bayes更适合）

## 6. Mini-Project

In this mini-project, we’ll tackle the exact same email author ID problem as the Naive Bayes mini-project, but now with an SVM

#### 6.1 Import, create, train and make predictions with the sklearn SVC classifier. When creating the classifier, use a linear kernel. What is the accuracy of the classifier?

## 7. Further exploration in SVM


```python

```
