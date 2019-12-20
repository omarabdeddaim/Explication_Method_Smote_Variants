import smote_variants as sv
import sklearn.datasets as datasets

dataset = datasets.load_wine()
X, y = dataset['data'], dataset['target']

oversampler = sv.MulticlassOversampling(sv.distance_SMOTE())

# X_samp and y_samp contain the oversampled dataset
X_samp, y_samp = oversampler.sample(X, y)
