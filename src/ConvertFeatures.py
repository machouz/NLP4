from datetime import datetime
import sys
import os
from utils import *

features_file = sys.argv[1] if len(sys.argv) > 1 else 'features_file'
feature_vecs_file = sys.argv[2] if len(sys.argv) > 2 else 'feature_vecs_file'
feature_map_file = sys.argv[3] if len(sys.argv) > 1 else 'feature_map_file'


def featureConvert(fname):
    features_id = {}
    i = 0
    data = []
    for line in file(fname):
        line = line[:-1].split(" ")
        if line[0] not in features_id:
            features_id[line[0]] = i
            i += 1
    for line in file(fname):
        line = line[:-1].split(" ")
        label = features_id[line[0]]
        features = []
        for feature in line[1:]:
            if feature not in features_id:
                features_id[feature] = i
                i += 1
            features.append(features_id[feature])

        features = map(lambda x: str(x) + ":1", sorted(features))

        vec = [str(label)] + features
        data.append(" ".join(vec))

    dic_to_file(features_id, feature_map_file)
    write_to_file(feature_vecs_file, data)
    return data


if __name__ == '__main__':
    start = datetime.now()
    data = featureConvert(features_file)
    print(datetime.now() - start)
