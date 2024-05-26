import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import sys

# Ignore warnings if there are no specific warning options set
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

print('Reading dataset')
file_path = 'data/data2.xlsx'
df = pd.read_excel(file_path)

# Print unique values in the 'Itemname' column
unique_values = df['Itemname'].unique()

# Drop rows with any missing values
if df.isna().sum().sum() > 0:
    df = df.dropna()

# Ensure the data types are correct
df['Price'] = df['Price'].astype('float64')
df['CustomerID'] = df['CustomerID'].astype('int')
df['Date'] = pd.to_datetime(df['Date'])
df['Itemname'] = df['Itemname'].str.strip()
df['Total_Price'] = df.Quantity * df.Price

df.info()
print(df.head())

# Define the hot_encode function for binary encoding
def hot_encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# Define the apriori_model function to run the Apriori algorithm
def apriori_model():
    data = df[df['Country'] >= 'United Kingdom']
    print('Number of transactions:', len(df))
    
    # Prepare the basket (transaction) dataset for Apriori without considering quantity
    apriori_df = data.copy()
    apriori_df['Quantity'] = apriori_df['Quantity'].apply(hot_encode)
    basket = (apriori_df.groupby(['BillNo', 'Itemname'])['Quantity']
              .max().unstack().reset_index().fillna(0)
              .set_index('BillNo'))
    print(basket.head())
    
    # Run the Apriori algorithm
    frq_items = apriori(basket, min_support=0.02, use_colnames=True)
    print('Number of frequent itemsets:', len(frq_items))
    
    # Generate association rules
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    print('Number of rules:', len(rules))
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    return rules

# Run the apriori_model function and print the resulting rules
rules = apriori_model()
print(rules.head())
# Remove rows where support is less than 0.2 and confidence is less than 0.5
rules_1 = rules[~((rules['support'] < 0.2) | (rules['confidence'] < 0.5))]
print(rules_1.head())

# Export the filtered rules to an Excel file
rules.to_excel('data/d4.xlsx', index=False)
