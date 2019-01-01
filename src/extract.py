import sys
from utils import *

if __name__ == '__main__':

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.DEV.processed.txt'
    output_file = sys.argv[2] if len(sys.argv) > 1 else 'data/Annotation/output_greedy'


    data = read_processed_file(input_file)
    output = []

    for num, sentence in data:
        persons = []
        locations = []
        for word in sentence:
            if word['TYPE'] == 'GPE':
                locations.append(word['TEXT'])
            elif word['TYPE'] == 'PERSON':
                persons.append(word['TEXT'])

        for per in persons:
            for loc in locations:
                annot = num + ' ' + per + ' Live_In ' + loc + ' (' + sentence + ')'
                output.append(annot)


    np.savetxt(output_file, output)
