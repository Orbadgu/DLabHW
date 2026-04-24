import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def train_hybrid_svm():
    print("Loading extracted features...")
    X_train = np.load('features/X_train.npy')
    y_train = np.load('features/y_train.npy')
    X_test = np.load('features/X_test.npy')
    y_test = np.load('features/y_test.npy')

    print("Training Linear SVM... (Optimized for large datasets)")
    clf = LinearSVC(C=0.1, max_iter=2000, dual=False)
    clf.fit(X_train, y_train)

    print("Testing SVM...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Hybrid Model (CNN + SVM) Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
