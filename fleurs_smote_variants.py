
import smote_variants as sv
import imbalanced_databases as imbd

dataset = imbd.load_iris0()
X, y = dataset['data'], dataset['target']

oversampler = sv.distance_SMOTE()

# X_samp and y_samp contain the oversampled dataset
X_samp, y_samp = oversampler.sample(X, y)
