import json
import pandas as pd
from collections import defaultdict

def get_label_dict():
    csv_fns = [f"/share/data/speech/hackathon_2022/data/slurp/csvs/{x}" for x in ["devel-type=direct.csv", "test-type=direct.csv", "train-type=direct.csv"]]
    output_fn = "downstream/slurp/class_dict.json"
    class2labels = defaultdict(list)
    class_names = ['scenario', 'action']
    for csv_fn in csv_fns:
        df = pd.read_csv(csv_fn)
        for semantic in df['semantics']:
            items = semantic.split('|')
            for i_class, class_name in enumerate(class_names):
                label = items[i_class].split(':')[1].strip()[1:-1]
                class2labels[class_name].append(label)
            # if class_name == 'scenario':
            #     label = items[0].split(':')[1].strip()[1:-1]
            # elif class_name == 'action':
            #     label = items[1].split(':')[1].strip()[1:-1]
            # labels.append(label)
    for class_name, labels in class2labels.items():
        labels = sorted(set(labels))
        # s2i = {label: i for i, label in enumerate(labels)}
        class2labels[class_name] = labels
    json.dump(class2labels, open(output_fn, 'w'), indent=4)
    return


if __name__ == '__main__':
    get_label_dict()
