from decisionTree import parse_label



def parse_file(file_path):
    map = {}
    with open(file_path) as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)
        for row in reader:
            


def calc_marginal_entropy(file_path):
    map = parse_label(input_file)
    majority = max(map.values())
    minority = min(map.values())
    total = float(majority + minority)
    entropy = -(minority/total*math.log(minority/total,2)+majority/total*math.log(majority/total,2))
    return entropy

def calc_conditional_entropy


