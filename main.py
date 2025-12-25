from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


clf = LogisticRegression()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_probs = clf.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
