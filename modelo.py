import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

zoo_data = pd.read_csv('arquivos-csv/zoo.csv')
class_data = pd.read_csv('arquivos-csv/class.csv')
class_mapping = dict(zip(class_data['Class_Number'], class_data['Class_Type']))

X = zoo_data.drop(['class_type', 'animal_name'], axis=1)
y = zoo_data['class_type']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf = clf.fit(X_treino, y_treino)

preditos = clf.predict(X_teste)
print("Acuracia:", accuracy_score(y_teste, preditos))

pickle.dump(clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
