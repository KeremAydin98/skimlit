from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_results(y_true, y_pred):

  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred,average='weighted')
  recall = recall_score(y_true, y_pred,average='weighted')
  f1 = f1_score(y_true, y_pred,average='weighted')

  results = {"accuracy":accuracy,
             "precision":precision,
             "recall":recall,
             "f1_score":f1}

  final_results = pd.DataFrame(results,index=["Model"])

  return final_results