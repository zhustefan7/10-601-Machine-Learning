from __future__ import print_function
import sys

def reverse(input_file_dir, output_file_dir):
    input_file = open(input_file_dir , 'r')
    output_file = open(output_file_dir , 'w')
    container = []
    for line in input_file:
        container.append(line)
    print(container)
    
    for i in range(len(container)-1,-1,-1):
        output_file.write(container[i])
    output_file.close()




if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    print ("The input file is : % s " % (infile))
    print ("The output file is : % s " % (outfile))
    reverse(infile , outfile)