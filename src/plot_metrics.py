import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Load trained model and test data
model = joblib.load("model.joblib")
X_test, y_test = joblib.load("test_data.joblib")

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')

# Save the image
plt.title("Confusion Matrix")
plt.savefig("metrics.png")
plt.close()

# Create markdown report
with open("report.md", "w") as f:
    f.write("# Model Report\n\n")
    f.write("## Confusion Matrix\n\n")
    f.write("![Confusion Matrix](./metrics.png)\n")
                
