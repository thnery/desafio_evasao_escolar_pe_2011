import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as sm

def read_data_from_csv(file_path):
  data = pd.read_csv(file_path, delimiter=';')
  return np.array(data)

def classify_with_SVC(file_path):
  data = read_data_from_csv(file_path)
  X, y = data[:, :5], data[:, 5]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

  # clf = SVC(kernel='linear', cache_size=2048)
  # clf = SVC(kernel='rbf', cache_size=2048)
  # clf = SVC(kernel='poly', cache_size=2048)
  clf = SVC(kernel='sigmoid', cache_size=2048)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(y_pred)
  score = clf.score(X_test, y_test)
  print(score)

  accuracy = sm.accuracy_score(y_test, y_pred)
  precision = sm.precision_score(y_test, y_pred)
  recall = sm.recall_score(y_test, y_pred)
  f1_score = sm.f1_score(y_test, y_pred)

  print("accuracy: %.4f precision: %.4f recall: %.4f f1-score: %.4f" % (accuracy, precision, recall, f1_score))


def main():
  classify_with_SVC("assets/desafio_dados_evasao_escolar_amostra13porcento.csv")

main()
