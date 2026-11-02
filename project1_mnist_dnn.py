"""
Project 1: Handwritten Digit Recognition using a Deep Neural Network (MNIST)
Lab Sheet-08 | MCA III Semester

Run:  python project1_mnist_dnn.py
Outputs: mnist_dnn.keras, mnist_learning_curves.png, mnist_confusion_matrix.png,
         mnist_comparison.csv
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------- 1. Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0   # normalise to [0, 1]
x_test = x_test.astype("float32") / 255.0
print("Train:", x_train.shape, "Test:", x_test.shape)

# ---------------------------------------------------------------- 2. Model
def build_model(units1=256, units2=128, dropout=0.3, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(units1, activation="relu"),   # ReLU: avoids vanishing gradient
        layers.BatchNormalization(),
        layers.Dropout(dropout),                   # regularisation
        layers.Dense(units2, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax"),    # 10-class probability output
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ------------------------------------------- 3. Small hyperparameter search
configs = [
    dict(units1=128, units2=64, dropout=0.2, lr=1e-3),
    dict(units1=256, units2=128, dropout=0.3, lr=1e-3),
    dict(units1=256, units2=128, dropout=0.3, lr=1e-2),
]
tuning = []
best_cfg, best_val = None, 0
for cfg in configs:
    m = build_model(**cfg)
    h = m.fit(x_train, y_train, validation_split=0.1, epochs=5,
              batch_size=128, verbose=0)
    v = h.history["val_accuracy"][-1]
    tuning.append({**cfg, "val_accuracy": round(v, 4)})
    print(cfg, "-> val_acc", round(v, 4))
    if v > best_val:
        best_val, best_cfg = v, cfg
print("Best config:", best_cfg)
pd.DataFrame(tuning).to_csv("mnist_hyperparameter_tuning.csv", index=False)

# ------------------------------------------------- 4. Train the final model
model = build_model(**best_cfg)
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                           restore_best_weights=True)]
t0 = time.time()
history = model.fit(x_train, y_train, validation_split=0.1, epochs=20,
                    batch_size=128, callbacks=callbacks, verbose=2)
dnn_time = time.time() - t0

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"DNN test accuracy: {test_acc:.4f}")

# ---------------------------------------------------- 5. Learning curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set(title="Loss", xlabel="Epoch", ylabel="Loss"); ax[1].legend()
plt.tight_layout(); plt.savefig("mnist_learning_curves.png", dpi=150); plt.close()

# ------------------------------------------------------ 6. Confusion matrix
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.title("Confusion Matrix (DNN)"); plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig("mnist_confusion_matrix.png", dpi=150); plt.close()

# ---------------------------------- 7. Save the trained model
model.save("mnist_dnn.keras")
print("Saved model -> mnist_dnn.keras")

# ------------------- 8. Compare with traditional ML (subset for speed)
xtr_flat = x_train.reshape(len(x_train), -1)
xte_flat = x_test.reshape(len(x_test), -1)
results = [{"model": "Deep Neural Network", "test_accuracy": test_acc,
            "train_time_s": dnn_time}]

t0 = time.time()
lr_model = LogisticRegression(max_iter=200).fit(xtr_flat[:20000], y_train[:20000])
results.append({"model": "Logistic Regression (20k samples)",
                "test_accuracy": accuracy_score(y_test, lr_model.predict(xte_flat)),
                "train_time_s": time.time() - t0})

t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
rf.fit(xtr_flat[:20000], y_train[:20000])
results.append({"model": "Random Forest (20k samples)",
                "test_accuracy": accuracy_score(y_test, rf.predict(xte_flat)),
                "train_time_s": time.time() - t0})

df = pd.DataFrame(results).round(4)
df.to_csv("mnist_comparison.csv", index=False)
print("\n=== Comparison: Traditional ML vs Deep Learning ===")
print(df.to_string(index=False))