import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def read_data_from_csv(file_path):
  data = pd.read_csv(file_path, delimiter=';')
  return np.array(data)

def classify_with_SVC(file_path):
  data = read_data_from_csv(file_path)
  X, y = data[:, :7], data[:, 7]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  print(y_test)

  clf = SVC(kernel='linear', cache_size=2048)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(y_pred)
  score = clf.score(X_test, y_test) * 100
  print(score)


def main():
  classify_with_SVC("assets/desafio_dados_evasao_escolar.csv")

main()
