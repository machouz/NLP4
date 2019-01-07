from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import pickle
import sys

feature_vecs_file = sys.argv[1] if len(sys.argv) > 1 else 'feature_vecs_file'
model_file = sys.argv[2] if len(sys.argv) > 2 else 'model'

X_train, Y_train = load_svmlight_file(feature_vecs_file)
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
pickle.dump(model, open(model_file, 'wb'))
