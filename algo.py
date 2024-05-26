import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

print('Reading dataset')
file_path = 'data/data2.xlsx'
df = pd.read_excel(file_path)
print(df.head())

unique_values = df['Itemname'].unique()
# print(len(unique_values))
# print(unique_values)

# print(df.dtypes)

if df.isna().sum().sum() > 0:
    df = df.dropna()

df['Price'] = df['Price'].astype('float64')
df['CustomerID'] = df['CustomerID'].astype('int')
df['Date'] = pd.to_datetime(df['Date'])
df['Itemname'] = df['Itemname'].str.strip()
df['Total_Price'] = df.Quantity * df.Price
df.info()

country = input(" Write the country of the customer: ")
ID = int(input(" Write the customer's ID number: "))

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
    
   
def apriori_model(country = country, ID = ID):
    data = df[df['Country'] == country]
    today_date = max(data["Date"])
    #RFM
    rfm = data.groupby('CustomerID').agg({'Date': lambda Date: (today_date - Date.max()).days,
                                     'CustomerID': lambda CustomerID: CustomerID.count(),
                                     'Total_Price': lambda Total_Price: Total_Price.sum()})
    rfm.columns = ["recency", "frequency", "monetary"]
    scaler = StandardScaler().fit(rfm)
    rfm_scale = scaler.transform(rfm)
    #Kmeans
    kmeans = KMeans(n_clusters = 4, n_init=25, max_iter=300)
    k_means = kmeans.fit(rfm_scale)
    segment = k_means.labels_
    rfm['segment'] = segment

    rfm = rfm.reset_index().rename(columns={'index': 'CustomerID'})
    new_df = data.merge(rfm, right_on = 'CustomerID', left_on = 'CustomerID')

    #Apriori

    number_of_cluster = list(rfm[rfm['CustomerID'] == ID]['segment'])[0]

    apriori_df = new_df[new_df['segment'] == number_of_cluster ]
    basket = (apriori_df.groupby(['BillNo', 'Itemname'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('BillNo'))
    # Encoding the datasets
    basket_encoded = basket.applymap(hot_encode)
    basket = basket_encoded

    frq_items = apriori(basket, min_support = 0.03, use_colnames = True)
    rules = association_rules(frq_items, metric ="lift", min_threshold = 0.8)
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
    return rules

rules = apriori_model(country=country, ID=ID)
print(rules.head())

rules.to_excel('data/rules.xlsx', index=False)










# import pandas as pd
# import sys
# from itertools import combinations

# def generate_subsets(lst):
#     all_subsets = []
#     for r in range(1, len(lst) + 1):
#         all_subsets.extend(list(combinations(lst, r)))
#     return all_subsets
# def find_present_combinations(s1, m):
#     present_combinations = []
#     all_subsets_s1 = generate_subsets(s1)
#     for subset in m:
#         subset_tuple = tuple(subset)
#         if subset_tuple in all_subsets_s1:
#             present_combinations.append(subset)
#     return present_combinations

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")

# print('Reading dataset')
# file_path = 'data/d4.xlsx'
# df = pd.read_excel(file_path)
# def frozenset_string_to_list(fset_string):
#     return list(eval(fset_string))
# df.drop(columns=['antecedent support','consequent support','leverage','conviction','zhangs_metric'], inplace=True)
# df['antecedents'] = df['antecedents'].apply(frozenset_string_to_list)
# df['consequents'] = df['consequents'].apply(frozenset_string_to_list)
# sl=[]
# consequents_list = []
# while True:
#     item=input("Enter item:")
#     # item= 'PINK REGENCY TEACUP AND SAUCER','GREEN REGENCY TEACUP AND SAUCER','GARDENERS KNEELING PAD CUP OF TEA'
#     sl.append(item)
#     print(sl)
#     master=[]
#     find_present_combinations(sl,master)

#     # For single item check
#     for index, row in df.iterrows():
#         if item in row['antecedents'][0]:
#             f='index is:'+str(index)+' row:'+str(row)
#             print(f)
#             print('<-',type(row['antecedents']),'->')
#             consequents_list.extend(row['consequents'])
    
#     if consequents_list:
#         print("Consequents for", sl, ":", list(set(consequents_list)))
#     else:
#         print("No consequents found for", item)
#     q=input("Press q to quit :")
#     if q=='q':
#         break