import os.path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import smote_variants as sv
import sklearn.datasets as datasets


cache_path = os.path.join(os.path.expanduser('~'), 'smote_test')

if not os.path.exists(cache_path):
    # L'ajout du chemin pour trouver la méthode smote_variants
    os.makedirs(cache_path)


dataset = datasets.load_breast_cancer()

dataset = {'data': dataset['data'],
           'target': dataset['target'], 'name': 'breast_cancer'}
# Renome les données par data et le target par le nom breast_cancer

knn_classifier = KNeighborsClassifier()  # Méthode des k plus proches voisins
dt_classifier = DecisionTreeClassifier()  # Méthode de L'arbre de décision

samp_obj, cl_obj = sv.model_selection(dataset=dataset,
                                      samplers=sv.get_n_quickest_oversamplers(
                                          5),
                                      classifiers=[
                                          knn_classifier, dt_classifier],
                                      cache_path=cache_path,
                                      n_jobs=5,
                                      max_samp_par_comb=35)

# génére donnée après sur-échantillonnage
X_samp, y_samp = samp_obj.sample(dataset['data'], dataset['target'])
cl_obj.fit(X_samp, y_samp)  # Pour voir l'état ou le diagnostique de notre KNN
