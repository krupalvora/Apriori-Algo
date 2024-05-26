import pandas as pd
import sys
from itertools import combinations

def generate_subsets(lst):
    all_subsets = []
    for r in range(1, len(lst) + 1):
        all_subsets.extend(list(combinations(lst, r)))
    return all_subsets

def frozenset_string_to_list(fset_string):
    return list(eval(fset_string))

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

print('Reading dataset')
file_path = 'data/d4.xlsx'
df = pd.read_excel(file_path)

df.drop(columns=['antecedent support', 'consequent support', 'leverage', 'conviction', 'zhangs_metric'], inplace=True)
df['antecedents'] = df['antecedents'].apply(frozenset_string_to_list)
df['consequents'] = df['consequents'].apply(frozenset_string_to_list)

sl = []
while True:
    item = input("Enter item: ")
    # item= 'PINK REGENCY TEACUP AND SAUCER','GREEN REGENCY TEACUP AND SAUCER','GARDENERS KNEELING PAD CUP OF TEA'
    sl.append(item)
    print("Current list of items:", sl)

    all_subsets_sl = generate_subsets(sl)
    present_combinations = []
    consequents_list = []

    for subset in all_subsets_sl:
        for index, row in df.iterrows():
            if set(subset).issubset(row['antecedents']):
                present_combinations.append(subset)
                consequents_list.extend(row['consequents'])

    if present_combinations:
        # print("Present combinations in antecedents:", present_combinations)
        print("Consequents for", sl, ":", list(set(consequents_list)-set(sl)))
    else:
        print("No consequents found for", item)

    q = input("Press q to quit: ")
    if q == 'q':
        break
