# TetanggaMasaGitu

TetanggaMasaGitu is a Python package that implements the K-Nearest Neighbors algorithm from scratch for classification. <br/>

### What is the K-Nearest Neighbor (KNN) Algorithm?
K-Nearest Neighbor (KNN) is a supervised learning algorithm used for both classification and regression. It is non-parametric, meaning it doesn’t make any assumptions about the underlying data distribution, which makes it versatile for various applications. KNN works by analyzing the proximity or “closeness” of data points based on specific distance metrics.


<div align="center">
    <img src="/docs/knn.png" alt="K-Nearest Neighbors"/>
</div>

### Steps

1. Calculate Distance (Euclidean Distance) <br/>
In this package, the Euclidean Distance formula is used to measure the similarity between the test data point and each training data point.
<div align="center">
    <img src="/docs/euc.png" alt="Euclidean Distance"/>
</div>

2. Sort the Distance <br/>
After calculating all distances, sort them in ascending order (from the smallest to the largest).

3. Voting <br/>
Select the top K neighbors and perform a majority vote based on their class labels.


### Implementation

In this example, i'm use the Iris dataset from the Scikit-learn library. <br/>
It contains 150 samples of iris flowers with four features each (sepal length, sepal width, petal length, and petal width), divided into three species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

<div align="center">
    <img src="/docs/iris-ds.png" alt="Iris Datasett"/>
</div>


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

# Separate features and target
X = df.iloc[:,:-1]
y = df.iloc[:, -1]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Fit and Predict
K = 3
my_model_knn = KNeighborsClassifier(k=K)
my_model_knn.fit(X_train, y_train)
y_pred_my_model = my_model_knn.predict(X_test)
print(y_pred_my_model)

# Accuracy TMG
print(f"Accuracy TMG: {accuracy_score(y_test, y_pred_my_model)}")

```

#### Accuracy Result TMG
```bash
Accuracy TMG: 1.0
```

### Comparison with Scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier

# Fit and Predict
knn_sklearn = KNeighborsClassifier(K)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)


```

#### Accuracy Result Sklearn
```bash
Accuracy Sklearn: 1.0
```

```python
print(np.array_equal(y_pred_my_model,y_pred_sklearn))
True
```

#### Code
- [Implementation on ipynb](knn.ipynb)
- [Package Code](/tetanggamasagitu/tmg_knn.py)

#### Source
- [What is the K-Nearest Neighbor (KNN) Algorithm?](https://www.appliedaicourse.com/blog/knn-algorithm-in-machine-learning/)
- [What is Euclidean Distance?](https://www.datacamp.com/tutorial/euclidean-distance)