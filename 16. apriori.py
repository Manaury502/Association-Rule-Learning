# Importar las libreri­as necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Importar el data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])

# Entrenar el algoritmo de Apriori
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2,
                min_lift = 3, min_length = 2)

# Funcion para inspeccionar los resultados y extraer los datos relevantes
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]  # Items en el lado izquierdo
    rhs = [tuple(result[2][0][1])[0] for result in results]  # Items en el lado derecho
    supports = [result[1] for result in results]  # Soporte
    confidences = [result[2][0][2] for result in results]  # Confianza
    lifts = [result[2][0][3] for result in results]  # Lift
    return list(zip(lhs, rhs, supports, confidences, lifts))

# Organizar los resultados en un DataFrame
results = list(rules)
results_in_df = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Mostrar los resultados no ordenados
print("Resultados no ordenados:")
print(results_in_df)

# Mostrar los resultados ordenados por Lift
print("\nTop 10 resultados por Lift:")
print(results_in_df.nlargest(10, 'Lift'))
