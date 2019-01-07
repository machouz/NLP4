from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import pickle
from sys import argv

feature_vecs_file = argv[1]
model_file = argv[2]

X_train, Y_train = load_svmlight_file(feature_vecs_file)
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
pickle.dump(model, open(model_file, 'wb'))
