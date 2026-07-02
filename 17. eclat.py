# Importar las librerías necesarias
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar el dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []

# Preprocesar las transacciones
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Convertir las transacciones a un DataFrame adecuado para usar en mlxtend
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar el algoritmo Apriori
frequent_itemsets = apriori(df, min_support=0.003, use_colnames=True)

# Generar reglas de asociacion a partir de los itemsets frecuentes
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1,num_itemsets=2)

# Mostrar las reglas
print(rules)
