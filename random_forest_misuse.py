import imblearn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from imblearn.combine import SMOTEENN
from collections import OrderedDict
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---Load the dataset
dataset = pd.read_csv("kddcup99_csv.csv")

print(dataset)

X = dataset.drop(['label'], axis = 1)
# ---Labeling
y = dataset['label']

# ---Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ---Label encode the categorical values and convert them to numbers 
le = LabelEncoder()
le.fit(X_train['protocol_type'].astype(str))
X_train['protocol_type'] = le.transform(X_train['protocol_type'].astype(str))
X_test['protocol_type'] = le.transform(X_test['protocol_type'].astype(str))

le.fit(X_train['service'].astype(str))
X_train['service'] = le.transform(X_train['service'].astype(str))
X_test['service'] = le.transform(X_test['service'].astype(str))

le.fit(X_train['flag'].astype(str))
X_train['flag'] = le.transform(X_train['flag'].astype(str))
X_test['flag'] = le.transform(X_test['flag'].astype(str))

# --- Optimization of Error Rate
RANDOM_STATE = 1

# ---- Generate a binary classification dataset.
X, y = make_classification(n_samples=3000, n_features=42,
                           n_clusters_per_class=5, n_informative=15,
                           random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# ---- Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# ---- Range of `n_estimators` values to explore.
min_estimators = 40
max_estimators = 200

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# ---- Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

# ---- Minority Intrusions Detection
X, y = make_classification(n_samples=3000, weights=[0.99], flip_y=0)
# summarize class distribution
print(Counter(y))
# define oversampling strategy
over = RandomOverSampler(sampling_strategy=0.1)
# fit and apply the transform
X, y = over.fit_resample(X, y)
# summarize class distribution
print(Counter(y))
# define undersampling strategy
under = RandomUnderSampler(sampling_strategy=0.5)
# fit and apply the transform
X, y = under.fit_resample(X, y)
# summarize class distribution
print(Counter(y))

# --- Feature Selection
fs = SelectFromModel(RandomForestClassifier(n_estimators = 100))
fs.fit(X_train, y_train)
fs.get_support()
selected_feat= X_train.columns[(fs.get_support())]
len(selected_feat)
print(selected_feat)
