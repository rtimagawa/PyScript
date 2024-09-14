import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Pré-Processamento de dados -----------------------------------------------------------------------

# Carregar os dados
df = pd.read_csv('dataset.csv')

# Verificar dados faltantes
#print(df.isnull().sum())

# Descrição dos dados
#print(df.describe())

# Separar as características e a variável alvo
x = df.drop('species',axis=1)  # Assume que a coluna 'species' é a variável alvo
y = df['species']


#Pré-Processamento de dados -----------------------------------------------------------------------
'\n'

#Treinamento com Técnicas de Aprendizado Supervisionado:

# Dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Treinar o modelo de Árvore de Decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)

# Fazer previsões
y_pred = clf.predict(x_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}\n')
print('Relatório de Classificação:\n')
print(classification_report(y_test, y_pred))

# Matriz de Confusão
print('\nMatriz de Confusão:\n')
print(confusion_matrix(y_test, y_pred))

'''
A acurácia de 100% e os valores de precisão, recall e F1-score também todos em 1.00.
Precisão: Proporção de verdadeiros positivos entre as previsões positivas. Aqui, todas as previsões para cada classe foram corretas.
Recall: Proporção de verdadeiros positivos em relação ao número total de amostras da classe. Como todos foram corretamente classificados, o recall também é 1.00 para todas as classes.
F1-score: A média harmônica entre precisão e recall, que neste caso também é perfeita.

A matriz mostra que o modelo classificou corretamente todos os exemplos de cada classe, sem erros. Por exemplo:
Todas as 19 amostras de Iris-setosa foram classificadas corretamente.
As 13 amostras de Iris-versicolor e Iris-virginica também foram classificadas sem erros.

'''