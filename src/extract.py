import sys
from utils import *

if __name__ == '__main__':

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/Processed_Corpus/Corpus.DEV.new.processed.txt'
    output_file = sys.argv[2] if len(sys.argv) > 1 else 'data/Annotation/output_greedy.txt'


    data = read_processed_file(input_file)
    output = []

    for num, sentence in data:
        persons = []
        locations = []
        i = 0
        while i < len(sentence):
            word = sentence[i]
            if sentence[i]['TYPE'] == 'GPE' or sentence[i]['TYPE'] == 'NORP' or sentence[i]['TYPE'] == 'LOC':
                pers = sentence[i]['TEXT']
                i += 1
                while i < len(sentence) and sentence[i]['IOB'] == 'I':
                    pers += ' ' + sentence[i]['TEXT']
                    i += 1
                locations.append(pers)
            elif sentence[i]['TYPE'] == 'PERSON':
                loca = sentence[i]['TEXT']
                i+= 1
                while i < len(sentence) and sentence[i]['IOB'] == 'I':
                    loca += ' ' + sentence[i]['TEXT']
                    i += 1
                persons.append(loca)

            else:
                i += 1

        for per in persons:
            for loc in locations:
                annot = num + '\t' + per + '\tLive_In\t' + loc + '\t(' + 'aa' + ')'
                output.append(annot)



    np.savetxt(output_file, output, fmt='%s')
