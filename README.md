# Powerful scikit-learn workflow
Sample project showing how to flexibly and efficiently train machine learning models with scikit-learn.


## Why is this useful?
Because during investigative prototyping of machine learning models, delegating appropriately to the framework can save hundreds of development hours and reduce the amount of code to maintain by thousands of lines.


## Training configuration
The code in train.py implements a multivariate classification experiment and illustrates how to apply feature-specific preprocessing,
```python
ColumnTransformer(transformers=[
    ('scaler', StandardScaler(), ['Insulin', 'BloodPressure']),
    ('passthrough', 'passthrough', ['Glucose'])
])
```
how to use a custom scoring metric,
```python
def roc_auc_without_uncertain_samples(y_true, y_score):
    uncertain = (
        (y_score > np.quantile(y_score, 0.45)) &
        (y_score < np.quantile(y_score, 0.55))
    )
    return roc_auc_score(y_true[~uncertain], y_score[~uncertain])
```
and how to optimize multiple models and corresponding hyperparameters in a single execution.
```python
[
    (
        RandomForestClassifier(random_state=123),
        {'bootstrap': [True, False], 'max_depth': [3, 20]}
    ),
    (
        SVC(probability=True, random_state=123),
        {'kernel': ['rbf', 'linear'], 'C': [0.1, 10]}
    )
]
```


## Workflow


### Get dataset
```
curl https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv \
    | sed s/Outcome/y/g > data.csv
```
This downloads the public diabetes dataset used to illustrate the workflow.

### Train model
```
python train.py
```
This runs model training and saves a `GridSearchCV` object in a file named search.joblib which contains detailed information about the execution.


### Inspect results
report.py shows how to inspect the object stored in search.joblib to acquire rich detail about the training execution, namely the winning model instance,
```python
search.best_estimator_
```
```
Pipeline(steps=[
    (
        'columntransformer',
        ColumnTransformer(transformers=[
            ('scaler', StandardScaler(), ['Insulin', 'BloodPressure']),
            ('passthrough', 'passthrough', ['Glucose'])
        ])
    ),
    (
        'classifier',
        RandomForestClassifier(max_depth=3, random_state=123)
    )
])
```

the hyperparameter optimization results,
```python
df = pd.DataFrame(search.cv_results_)
model2str = lambda model: type(model).__name__
df['param_classifier'] = df['param_classifier'].apply(model2str)
df.filter(regex='param_|mean_test_selection-criterion')
```
|    | param_classifier       |   param_classifier__bootstrap |   param_classifier__max_depth |   param_classifier__C | param_classifier__kernel   |   mean_test_selection-criterion |
|---:|:-----------------------|------------------------------:|------------------------------:|----------------------:|:---------------------------|--------------------------------:|
|  0 | RandomForestClassifier |                             1 |                             3 |                 nan   | nan                        |                        0.786414 |
|  1 | RandomForestClassifier |                             1 |                            20 |                 nan   | nan                        |                        0.743816 |
|  2 | RandomForestClassifier |                             0 |                             3 |                 nan   | nan                        |                        0.777103 |
|  3 | RandomForestClassifier |                             0 |                            20 |                 nan   | nan                        |                        0.696485 |
|  4 | SVC                    |                           nan |                           nan |                   0.1 | rbf                        |                        0.804755 |
|  5 | SVC                    |                           nan |                           nan |                   0.1 | linear                     |                        0.798838 |
|  6 | SVC                    |                           nan |                           nan |                  10   | rbf                        |                        0.779612 |
|  7 | SVC                    |                           nan |                           nan |                  10   | linear                     |                        0.799157 |

and what cross validation folds were used.
```python
[test_idx for train_idx, test_idx in search.cv]
```
```
[array([  0,   3,   6, ... ]), array([  1,  14,  15, ... ]), array([  2,   4,   5, ... ])]
```


### Generate predictions
predict.py shows how to generate predictions, for illustrative purposes using the training data.


## On custom and third-party modules
For projects that require models or preprocessing modules that are implemented in-house or as part of other frameworks, it is often a viable strategy to [wrap them inside scikit-learn objects](https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator).


## Dependency versions used for development
- Python 3.8.8
- scikit-learn 0.24.2
- pandas 1.2.3
