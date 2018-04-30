---
title: "Building Scikit-Learn Pipelines With Pandas DataFrames"
date: 2018-04-16T08:10:13-05:00
categories:
- Python
- Pandas
- Scikit-Learn
- Machine Learning
comments: true
---

I've used [scikit-learn](http://scikit-learn.org/) for a number of years now. Although it is a useful tool for building machine learning pipelines, I find it difficult and frustrating to integrate `scikit-learn` with [pandas](https://pandas.pydata.org/) DataFrames, especially in production code. Why? Because `scikit-learn`:

* Operates on `numpy` matrices, not `DataFrames`
* Does not preserve feature names
* Omits column data types
* Pretty much ignores anything useful provided by `pandas`.

Of course, [a DataFrame is a numpy array](https://medium.com/@ericvanrees/pandas-series-objects-and-numpy-arrays-15dfe05919d7) with some extra sugar for data manipulation. But that sugar helps identify column types, keeps track of feature names when calculating feature importance, etc. What's worse is that categorical features are consistently a PITA because they [must be encoded](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) first and often require separate processing rules, such as imputing with the most frequent value. And finally, to close out my scikit rant, there's no simple way to filter unnecessary columns from a `DataFrame` before a model is built.

In this post, my goal is to share with you an approach to build a scikit-learn [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) that simplifies the integration with a pandas DataFrame. We're going to construct a `Pipeline` and:

* Apply an `ColumnSelector` to filter a `DataFrame` to the appropriate columns
* Construct a `TypeSelector` to apply separate operations to numerical and categorical features
* Build a `scikit-learn` pipeline using the transformed `DataFrame`

For our `Pipeline`, let's use the `churn` binary classification data set from the [Penn Machine Learning Benchmarks](https://github.com/EpistasisLab/penn-ml-benchmarks). The [data set's details](https://github.com/EpistasisLab/penn-ml-benchmarks/blob/master/datasets/classification/classification_datasets_pmlb.tsv) are sparse, but we know the following:

* 5000 observations
* Class labels consist of 4293 0's and 707 1's
* 15 numeric features
* 2 binary features
* 2 categorical features

After loading the data set below, we will:

* Set the appropriate column data types (important for our downstream `Pipeline`)
* Update 500 random items as missing values to demonstrate imputation in the `Pipeline`
* Partition the data set into a training set (2/3 of the observations) and a test data set (remaining 1/3)


```python
import pandas as pd
import pmlb

df = pmlb.fetch_data('churn', return_X_y=False)

# Remove the target column and the phone number
x_cols = [c for c in df if c not in ["target", "phone number"]]

binary_features = ["international plan", "voice mail plan"]
categorical_features = ["state", "area code"]

# Column types are defaulted to floats
X = (
    df
    .drop(["target"], axis=1)
    .astype(float)
)
X[binary_features] = X[binary_features].astype("bool")

# Categorical features can't be set all at once
for f in categorical_features:
    X[f] = X[f].astype("category")

y = df.target

# Randomly set 500 items as missing values
random.seed(42)
num_missing = 500
indices = [(row, col) for row in range(X.shape[0]) for col in range(X.shape[1])]
for row, col in random.sample(indices, num_missing):
    X.iat[row, col] = np.nan

# Partition data set into training/test split (2 to 1 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=42)
```

# Column Selector

Generally speaking, large Excel spreadsheets, CSV files, and other tabular data sets contain a large number of columns. Often, only some of those columns are useful. That's why we need a column selector so that we can:

* Ignore unimportant features from someone else's spreadsheet
* Use columns chosen by a feature-selection method
* Omit columns with little variance or excessive sparsity
* Skip features with target leak
* Test a model's performance using a subset of important features identified by an expert

While we're still talking column selection, you might ask whether we couldn't just use pandas to filter out the column. Most certainly, you can, but often in a `Pipeline`, you want to ensure your data set contains only those features you have specified manually, from a ETL microservice, etc. This is especially true in production ML systems, where, for example, an unwanted column can lead to exceptions being thrown when a user requests a prediction.

Returning to the `churn` data set, we see a `phone number` feature. While the last 4 digits of a phone number may contain some information, we'll consider it a nonessential feature for our purposes. To demonstrate the `ColumnSelector`, let's select a handful of relevant columns.


```python
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
            
cs = ColumnSelector(columns=["state", "account length", "area code"])
cs.fit_transform(df).head()
```

|   state |   account length |   area code |
|--------:|-----------------:|------------:|
|      16 |              128 |         415 |
|      35 |              107 |         415 |
|      31 |              137 |         415 |
|      35 |               84 |         408 |
|      36 |               75 |         415 |

# Type Selector

One of my favorite components of a scikit-learn `Pipeline` is the [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html). Here's its definition from the docs:

> Concatenates results of multiple transformer objects.
>
> This estimator applies a list of transformer objects in parallel to the input data, then concatenates the results. This is useful to combine several feature extraction mechanisms into a single transformer.

To be honest, I just didn't see the `FeatureUnion`'s potential at first. But once I realized that a `FeatureUnion` allows us to process sets of features in parallel, my mind was blown.

What we're going to do is combine a `FeatureUnion` and a `TypeSelector` to apply transformations to the pandas columns based on their data type. To numerical features, we can imputing missing values before centering and scaling the features. To categorical features, we can [encode them](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) with a `OneHotEncoder` or similar before applying categorical-specific transformations. If we wish to handle other data types in specific ways, such as binary features or `datetime`, the `TypeSelector` will allow us to do that. If you want to see the data types of each column in a `DataFrame`, check out [`df.select_dtypes()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html).

As you can see, the implementation of the `TypeSelector` (based on [this gist](https://gist.github.com/StevenReitsma/b6e067ef1784ffae4d97f395f5f9d889#file-blogpost-pandas3-py)) is pretty simple.


```python
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
```

Let's apply the `TypeSelector` to select only the categorical features from the `churn` data set.


```python
ts = TypeSelector("category")
ts.fit_transform(X).head()
```

|   state |   area code |
|--------:|------------:|
|  16 | 415 |
|  35 | 415 |
|  31 | 415 |
|  35 | 408 |
|  36 | 415 |


# Preprocessing Pipeline

Now that we've walked through the `ColumnSelector` and `TypeSelector` transformers, let's build a preprocessing transformer to:

1. Select only the relevant feature columns (omits the `phone number` column)
2. Impute and standardize the numeric features
3. Impute and one-hot encode the categorical features
4. Impute the boolean features
5. Apply a `FeatureUnion` to join the transformed features into a single data set


```python
preprocess_pipeline = make_pipeline(
    ColumnSelector(columns=x_cols),
    FeatureUnion(transformer_list=[
        ("numeric_features", make_pipeline(
            TypeSelector(np.number),
            Imputer(strategy="median"),
            StandardScaler()
        )),
        ("categorical_features", make_pipeline(
            TypeSelector("category"),
            Imputer(strategy="most_frequent"),
            OneHotEncoder()
        )),
        ("boolean_features", make_pipeline(
            TypeSelector("bool"),
            Imputer(strategy="most_frequent")
        ))
    ])
)
```

Simple, huh? What I like about this preprocessing pipeline is that it's not only simple, but easy to read.

So how do we apply it to our training and test data sets? Easy enough.


```python
preprocess_pipeline.fit(X_train)

X_test_transformed = preprocess_pipeline.transform(X_test)
X_test_transformed

    <1667x71 sparse matrix of type '<class 'numpy.float64'>'
    	with 28955 stored elements in Compressed Sparse Row format>
```

Notice that the test data set is now a sparse matrix with 1,667 rows and 71 columns. That means, the test data set is ready for classification by a scikit-learn classifier. In case you're curious, the large increase in columns from 19 to 71 columns is the result of the `OneHotEncoder` we applied to the categorical features.

# Preprocessing Pipeline + Classifier + Grid Search

You'll notice that the transformation pipeline above is quite general. Most `Pipelines`, whether they be classification or regression problems, require applying operations to numeric, categorical, and boolean features. So we can use this preprocessing pipeline pattern again and again.

Here's the best part though. Combining the preprocessing pipeline with a classifier to classify `churn` is so easy.


```python
classifier_pipeline = make_pipeline(
    preprocess_pipeline,
    SVC(kernel="rbf", random_state=42)
)
```

The classifier pipeline uses a [support vector machine](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) with a [radial basis function](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) as its kernel. Currently, the pipeline applies the default hyperparameters. Instead of the defaults, let's search for the optimal `gamma` hyperparameter via 10-fold crossvalidation and [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). As a side note, one nice byproduct of our `Pipeline` is that the imputation and scaling is applied during each cross-validation fold. In doing so, we've avoided subtle overfitting that can occur if we're not careful.

To specify the hyperparameter grid, we use a dictionary. The keys indicate the hyperparameter to optimize, and the values give the candidate values. The key uses a `__` (two underscores) to delineate the classifier and the hyperparameter. If you're unfamiliar, you can chain multiple sets of `__` together in more complex pipelines: see [the Pipeline docs](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) for details.


```python
param_grid = {
    "svc__gamma": [0.1 * x for x in range(1, 6)]
}

classifier_model = GridSearchCV(classifier_pipeline, param_grid, cv=10)
classifier_model.fit(X_train, y_train)
```

Okay, we've finally trained our classifier. So how well did the classifier do? Let's check the ROC curve, which had an AUC of 0.912.


```python
y_score = classifier_model.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

# Plot ROC curve
plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)', size=16)
plt.ylabel('True Positive Rate (Sensitivity)', size=16)
plt.title('ROC Curve', size=20)
plt.legend(fontsize=14);
```

![roc_curve](https://user-images.githubusercontent.com/261183/39414312-9eb51f92-4bfb-11e8-8770-2619b8f81bc2.png)


And now, to make the pipeline copy/paste friendly, here’s the entire code block:


```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC

import numpy as np
import pandas as pd

from pmlb import fetch_data

import random

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re

sns.set(rc={"figure.figsize": (12, 8)})


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])



df = pmlb.fetch_data('churn', return_X_y=False)

# Remove the target column and the phone number
x_cols = [c for c in df if c not in ["target", "phone number"]]

binary_features = ["international plan", "voice mail plan"]
categorical_features = ["state", "area code"]

# Column types are defaulted to floats
X = (
    df
    .drop(["target"], axis=1)
    .astype(float)
)
X[binary_features] = X[binary_features].astype("bool")

# Categorical features can't be set all at once
for f in categorical_features:
    X[f] = X[f].astype("category")

y = df.target

# Randomly set 500 items as missing values
random.seed(42)
num_missing = 500
indices = [(row, col) for row in range(X.shape[0]) for col in range(X.shape[1])]
for row, col in random.sample(indices, num_missing):
    X.iat[row, col] = np.nan

# Partition data set into training/test split (2 to 1 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=42)


preprocess_pipeline = make_pipeline(
    ColumnSelector(columns=x_cols),
    FeatureUnion(transformer_list=[
        ("numeric_features", make_pipeline(
            TypeSelector(np.number),
            Imputer(strategy="median"),
            StandardScaler()
        )),
        ("categorical_features", make_pipeline(
            TypeSelector("category"),
            Imputer(strategy="most_frequent"),
            OneHotEncoder()
        )),
        ("boolean_features", make_pipeline(
            TypeSelector("bool"),
            Imputer(strategy="most_frequent")
        ))
    ])
)

classifier_pipeline = make_pipeline(
    preprocess_pipeline,
    SVC(kernel="rbf", random_state=42)
)

param_grid = {
    "svc__gamma": [0.1 * x for x in range(1, 6)]
}

classifier_model = GridSearchCV(classifier_pipeline, param_grid, cv=10)
classifier_model.fit(X_train, y_train)

y_score = classifier_model.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

# Plot ROC curve
plt.figure(figsize=(16, 12))
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)', size=16)
plt.ylabel('True Positive Rate (Sensitivity)', size=16)
plt.title('ROC Curve', size=20)
plt.legend(fontsize=14);
```
