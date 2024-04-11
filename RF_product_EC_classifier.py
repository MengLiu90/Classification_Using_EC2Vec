import pandas as pd
import numpy as np
import joblib
from numpy import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv('embedded_products_ECs_data.csv')
X = df.drop(columns=['product', 'EC', 'label'])  # Features
y = df['label']  # Target label

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=30,
                                       min_samples_leaf=4, min_samples_split=5)

rf_classifier.fit(X_train, y_train)
joblib.dump(rf_classifier, 'product_ec_classifier_trained_model.pkl')

# # load the trained model
# rf_classifier = joblib.load('product_ec_classifier_trained_model.pkl')

# Evaluate the Model
y_pred = rf_classifier.predict(X_test)

test_acc = metrics.accuracy_score(y_test, y_pred)

y_score = rf_classifier.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, 1], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
confusionmatrix = metrics.confusion_matrix(y_test, y_pred)
p_score = metrics.precision_score(y_test, y_pred)
r_score = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)

TP = confusionmatrix[1][1]
TN = confusionmatrix[0][0]
FP = confusionmatrix[0][1]
FN = confusionmatrix[1][0]
fpr_fromCF = FP / (FP + TN)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print('accuracy', test_acc)
print('auc', roc_auc)
print('confusion matrix', confusionmatrix)
print('precision', p_score)
print('recall', r_score)
print('f1', f1)
print('balanced accuracy', bal_acc)
print('FPR', fpr_fromCF)
print('mcc', mcc)
print(classification_report(y_test, y_pred))