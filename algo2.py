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

# country = input(" Write the country of the customer: ")
# ID = int(input(" Write the customer's ID number: "))

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
    
   
def apriori_model():
    data = df
    today_date = max(data["Date"])
    #RFM    
    rfm = data.groupby('CustomerID').agg({'Date': lambda Date: (today_date - Date.max()).days,
                                     'CustomerID': lambda CustomerID: CustomerID.count(),
                                     'Total_Price': lambda Total_Price: Total_Price.sum()})
    rfm.columns = ["recency", "frequency", "monetary"]
    rfm.head()
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

    number_of_cluster = list(rfm['segment'])[0]
    print(number_of_cluster)
    # print(len(number_of_cluster))
    apriori_df = new_df
    basket = (apriori_df.groupby(['BillNo', 'Itemname'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('BillNo'))
    # Encoding the datasets
    basket_encoded = basket.applymap(hot_encode)
    basket = basket_encoded
    print('basket:',len(basket))
    frq_items = apriori(basket, min_support = 0.02, use_colnames = True)
    print('frq_items:',len(frq_items))
    rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
    print('rules:',len(rules))
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
    return rules

rules = apriori_model()
print(rules.head())

# rules.to_excel('data/rules2.xlsx', index=False)