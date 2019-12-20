import smote_variants as sv
# pip install imblearn we shoud instal this package
import imblearn.datasets as imb_datasets

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

libras = imb_datasets.fetch_datasets()['libras_move']
X, y = libras['data'], libras['target']

oversampler = sv.MulticlassOversampling(sv.distance_SMOTE())
classifier = KNeighborsClassifier(n_neighbors=5)

# Constructing a pipeline which contains oversampling and classification as the last step.
model = Pipeline([('scale', StandardScaler()),
                  ('clf', sv.OversamplingClassifier(oversampler, classifier))])

model.fit(X, y)
param_grid = {'clf__oversampler': [sv.distance_SMOTE(proportion=0.5),
                                   sv.distance_SMOTE(proportion=1.0),
                                   sv.distance_SMOTE(proportion=1.5)]}

# Specifying the gridsearch for model selection
grid = GridSearchCV(model, param_grid=param_grid, cv=3,
                    n_jobs=1, verbose=2, scoring='accuracy')

# Fitting the pipeline
grid.fit(X, y)
