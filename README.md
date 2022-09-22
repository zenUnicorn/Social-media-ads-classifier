# Social media classifier
Machine Learning for Classifying Social Media Ads

Installation and importing of the required libraries

```python
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
```

### Dataset
The [dataset](https://medium.com/r/?url=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fd4rklucif3r%2Fsocial-network-ads) we are using for the purpose of classifying social media ads is acquired from Kaggle.

```python
df = pd.read_csv("Social_Network_Ads.csv")
df.head()
```

```python
df.describe()
df.isnull().sum()
```

### Dataset trends

```python
plt.figure(figsize=(13, 8))
plt.title("Product Bought by Individuals through Social Media Marketing")
sns.histplot(data=df, x="Age", hue="Purchased")
plt.show()
```

```python
plt.figure(figsize=(13, 8))
plt.title("Product Purchased by Individuals Depending on Income")
sns.histplot(data=df, x="EstimatedSalary", hue="Purchased")
plt.show()
```

### The classification model

```python
x = np.array(df[["Age", "EstimatedSalary"]])
y = np.array(df[["Purchased"]])
```

### Train test split

```python
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
```
The test_size is 10%.

### Let's last have a look at the model's classification report.

```python
print(classification_report(ytest, predictions))
```






