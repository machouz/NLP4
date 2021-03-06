import pickle
import sys
from utils import *
from ExtractFeatures import extract_features, extract_persons_location

print('------------------------------------------')
sys.stdout.flush()

def feature_convert(features_dic):
    features_vec = zero_features_vector.copy()
    for key, value in features_dic.items():
        if str(key) + "=" + str(value) in features2id:
            feature = features2id["{}={}".format(key,value)]
            features_vec[feature - 1] = 1
    return features_vec


input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.TRAIN.processed.txt'
model_file = 'model'
feature_map_file = 'feature_map_file'
output_file = sys.argv[4] if len(sys.argv) > 4 else 'data/Annotation/output_model_train.txt'

model = pickle.load(open(model_file, 'rb'))
features2id = file_to_dic(feature_map_file)
id2features = {v: k for k, v in features2id.items()}
zero_features_vector = np.zeros(max(features2id.values()))


with open(input_file, 'r') as f:
    first_line = f.readline()




data = read_processed_file(input_file)

sentences = []

for num, sentence in data:
    sent = extract_persons_location(num, sentence)
    sentences.extend(sent)

featured_data = []
output = []
for num, per, loc, sentence in sentences:
    features_dic = extract_features(num, per, loc, sentence)
    features_vec = feature_convert(features_dic)
    tag_index = model.predict([features_vec])[0]
    if tag_index == features2id['1']:
        output.append(num + '\t' + per["TEXT"] + '\tLive_In\t' + loc["TEXT"] + '\t(' + "aaa" + ')')
    np.savetxt(output_file, output, fmt='%s')
