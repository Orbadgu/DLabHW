import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.neural_network import MLPClassifier

from src.data import load_and_preprocess_data
from src.mlp import CustomMLP


def main():
    print("Veri yükleniyor ve ön işleme yapılıyor...")

    X_train, y_train, X_dev, y_dev, X_test, y_test = load_and_preprocess_data("Mall_Customers.csv")

    print(f"Eğitim Seti: {X_train.shape}, Dev Seti: {X_dev.shape}, Test Seti: {X_test.shape}\n")

    hidden_layer_sizes = [4, 8, 16]
    steps_list = [1000, 3000, 5000]  # Adımları artırdık
    best_acc = 0
    best_model = None
    best_params = {}

    print("--- Model Seçim Süreci (Grid Search) ---")
    for h_size in hidden_layer_sizes:
        for steps in steps_list:
            model = CustomMLP(input_size=X_train.shape[1],
                              hidden_size=h_size,
                              n_steps=steps,
                              learning_rate=0.5,
                              lambd=0.5)

            model.fit(X_train, y_train, X_val=X_dev, y_val=y_dev, verbose=False)

            preds_dev = model.predict(X_dev)
            acc = accuracy_score(y_dev, preds_dev)

            print(f"Hidden Size: {h_size:2d} | Steps: {steps:4d} | Dev Acc: {acc:.4f}")

            if acc > best_acc or (acc == best_acc and steps < best_params.get('n_steps', float('inf'))):
                best_acc = acc
                best_model = model
                best_params = {'hidden_size': h_size, 'n_steps': steps}

    print(f"\nSeçilen En İyi Model Parametreleri: {best_params} (Dev Accuracy: {best_acc:.4f})")

    preds_test = best_model.predict(X_test)
    print("\n--- TEST SETİ SONUÇLARI (Sıfırdan Yazılan MLP) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds_test):.4f}")
    print(f"Precision: {precision_score(y_test, preds_test, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, preds_test, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds_test, zero_division=0):.4f}")
    print("\nKarmaşıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, preds_test))

    print("\n--- SCIKIT-LEARN MLPClassifier KARŞILAŞTIRMASI ---")
    sklearn_mlp = MLPClassifier(hidden_layer_sizes=(best_params['hidden_size'],),
                                activation='logistic',
                                solver='sgd',
                                learning_rate_init=0.05,
                                max_iter=best_params['n_steps'],
                                random_state=42)

    sklearn_mlp.fit(X_train, y_train.ravel())
    sklearn_preds = sklearn_mlp.predict(X_test)

    print(f"Scikit-Learn Accuracy: {accuracy_score(y_test, sklearn_preds):.4f}")
    print("\nScikit-Learn Classification Report:")
    print(classification_report(y_test, sklearn_preds, zero_division=0))

    plt.plot(best_model.history['train_loss'], label='Train Loss')
    plt.plot(best_model.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve (Loss vs. Epochs)')
    plt.xlabel('Steps')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()