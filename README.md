# TetanggaMasaGitu

TetanggaMasaGitu is a Python package that implements the K-Nearest Neighbors algorithm from scratch for classification. <br/>

### What is the K-Nearest Neighbor (KNN) Algorithm?
K-Nearest Neighbor (KNN) is a supervised learning algorithm used for both classification and regression. It is non-parametric, meaning it doesn’t make any assumptions about the underlying data distribution, which makes it versatile for various applications. KNN works by analyzing the proximity or “closeness” of data points based on specific distance metrics.


<div align="center">
    <img src="/docs/knn.png" alt="K-Nearest Neighbors"/>
</div>

### Code

```python
from tetanggamasagitu import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Load dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Separate x and y
X = df.iloc[:,:-1]
y = df.iloc[:, -1]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Fit and Predict
K = 3
my_model_knn = KNeighborsClassifier(k=K)
my_model_knn.fit(X_train, y_train)
y_pred_my_model = my_model_knn.predict(X_test)
print(y_pred_my_model)


```



#### Source
- [What is the K-Nearest Neighbor (KNN) Algorithm?](https://www.appliedaicourse.com/blog/knn-algorithm-in-machine-learning/)
