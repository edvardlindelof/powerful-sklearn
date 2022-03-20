import pandas as pd
pd.set_option('max_columns', 300)
pd.set_option('display.width', 10000)
from joblib import load

# import necessary for deserialization
from train import roc_auc_without_uncertain_samples


search = load('search.joblib')

print()
print('winning model:')
print(search.best_estimator_)

df = pd.DataFrame(search.cv_results_)
model2str = lambda model: type(model).__name__
df['param_classifier'] = df['param_classifier'].apply(model2str)
print()
print(df.filter(regex='param_|mean_test_selection-criterion'))

print()
print('fold test indices:')
print([test_idx for train_idx, test_idx in search.cv])

