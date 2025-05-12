from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


Xgboost_model = joblib.load('xgboost_model.pkl')
x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl')


def evaluate_model(model, x_test, y_test, model_name="Model"):
   
   # Predict
   y_pred = model.predict(x_test)
   
   # Classification Report
   print(f"\nClassification Report for {model_name}:\n")
   print(classification_report(y_test, y_pred))
   
   # Confusion Matrix
   disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, cmap='Blues')
   disp.ax_.set_title(f"{model_name} - Confusion Matrix")
   plt.show()


evaluate_model(Xgboost_model , x_test , y_test ,'XGboost')