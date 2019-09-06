import csv
import math
import sys

def parse_label(file_path):
    map = {}
    with open(file_path) as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)
        print(headers)
        for row in reader:
            label = row[-1]
            if label not in map.keys():
                map[label] = 1
            else:
                map[label] +=1
    return map
            


             
def inspect(input_file, output_file):
    map = parse_label(input_file)
    majority = max(map.values())
    minority = min(map.values())
    total = float(majority + minority)
    error_rate = majority/total
    entropy = -(minority/total*math.log(minority/total,2)+majority/total*math.log(majority/total,2))

    output_file = open(output_file, 'w')
    output_file.write('entropy: %s' %str(entropy)+'\n')
    output_file.write('error_rate: %s' %str(error_rate))






if __name__ =="__main__":
    # infile = sys.argv[1]
    # outfile = sys.argv[2]
    # print ("The input file is : % s " % (infile))
    # print ("The output file is : % s " % (outfile))
    # inspect(infile , outfile)
    inspect('handout/politicians_train.tsv' , 'politician_inspect.txt')