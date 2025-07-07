from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

transactions = [
    ['susu', 'roti', 'mentega'],
    ['roti', 'bir', 'popok', 'telur'],
    ['susu', 'popok', 'bir', 'cola'],
    ['susu', 'roti', 'popok', 'bir'],
    ['roti', 'susu', 'popok'],
    ['susu', 'roti', 'keju'],
    ['roti', 'keju', 'bir'],
    ['roti', 'susu', 'keju', 'popok'],
    ['roti', 'keju', 'susu'],
    ['roti', 'susu', 'popok']
]

# Encode
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# Apriori
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)

# Association Rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
