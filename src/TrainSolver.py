from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sklearn
import pickle
import sys

feature_vecs_file = sys.argv[1] if len(sys.argv) > 1 else 'feature_vecs_file'
model_file = sys.argv[2] if len(sys.argv) > 2 else 'model'

X_train, Y_train = load_svmlight_file(feature_vecs_file)
model = sklearn.svm.LinearSVC(loss='hinge', max_iter=100000)
model.fit(X_train, Y_train)
# save the model to disk
pickle.dump(model, open(model_file, 'wb'))
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/ExtractFeatures.py', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/ConvertFeatures.py', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/TrainSolver.py', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/main.py', args='data/Processed_Corpus/Corpus.TRAIN.processed.txt model feature_map_file data/Annotation/output_model_train.txt', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/eval.py', args='data/Annotation/TRAIN.annotations.txt data/Annotation/output_model_train.txt', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')


# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/ExtractFeatures.py', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/ConvertFeatures.py', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/TrainSolver.py', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/main.py', args='data/Processed_Corpus/Corpus.DEV.processed.txt model feature_map_file data/Annotation/output_model_dev.txt', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
# runfile('/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4/src/eval.py', args='data/Annotation/DEV.annotations.txt data/Annotation/output_model_dev.txt', wdir='/Users/machou/Documents/Machon Lev/Chana 5/Semestre 1/NLP/Ass4')
