import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import SCORERS as skl_scorers
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from joblib import dump


df = pd.read_csv('data.csv')
X = df.drop(columns='y')
y = df['y']


def roc_auc_without_uncertain_samples(y_true, y_score):
    uncertain = (
        (y_score > np.quantile(y_score, 0.45)) &
        (y_score < np.quantile(y_score, 0.55))
    )
    return roc_auc_score(y_true[~uncertain], y_score[~uncertain])

SCORING = {
    'roc-auc': skl_scorers['roc_auc'],
    'selection-criterion': make_scorer(
        roc_auc_without_uncertain_samples, needs_threshold=True
    )
}

SPLITTER = StratifiedKFold(3, shuffle=True, random_state=123)

MODELS = [
    (
        RandomForestClassifier(random_state=123),
        {'bootstrap': [True, False], 'max_depth': [3, 20]}
    ),
    (
        SVC(probability=True, random_state=123),
        {'kernel': ['rbf', 'linear'], 'C': [0.1, 10]}
    )
]


pipeline = Pipeline(steps=[
    ('columntransformer', ColumnTransformer([
        ('scaler', StandardScaler(), ['Insulin', 'BloodPressure']),
        ('passthrough', 'passthrough', ['Glucose'])
    ])),
    ('classifier', None)
])

grid = [
    {
        'classifier': [clf],
        **{
            f'classifier__{param}': values
            for param, values in clf_grid.items()
        }
    }
    for clf, clf_grid in MODELS
]

cv = list(SPLITTER.split(X, y))
search = GridSearchCV(
    pipeline,
    grid,
    scoring=SCORING,
    cv=cv,
    refit='selection-criterion'
)
search.fit(X, y)
dump(search, 'search.joblib')
