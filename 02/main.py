from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

compound1 = [1, 1, 1]
compound2 = [0, 0, 0]
compound3 = [1, 0, 0]
compound4 = [0, 1, 0]
compound5 = [1, 0, 0]
compound6 = [0, 0, 1]

data = [compound1, compound2, compound3, compound4, compound5, compound6]
label = ['S', 'N', 'S', 'N', 'S', 'S']

modelo = LinearSVC()
modelo.fit(data, label)

test1 = [1, 0, 0]
test2 = [0, 1, 1]
test3 = [1, 0, 1]

dados_teste = [test1, test2, test3]
expected_label = ['S', 'S', 'S']

predicts = modelo.predict(dados_teste)
actual_accuracy = accuracy_score(expected_label, predicts)

print("Accuracy: %.2f%%" % (actual_accuracy * 100))
