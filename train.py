"""Run: python train.py  (Telco-Customer-Churn.csv isi folder mein rakho)"""
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from churn_utils import NUM_COLS, ONEHOT_COLS, PASSTHROUGH_COLS, clean

df = pd.read_csv("Telco-Customer-Churn.csv")
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn", "customerID"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLS),
    ("bin", "passthrough", PASSTHROUGH_COLS),
])

model = Pipeline([
    ("clean", FunctionTransformer(clean)),
    ("prep", preprocess),
    ("smote", SMOTE(random_state=42)),        # sirf fit ke waqt chalta hai, predict pe skip
    ("svc", SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)),
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Recall  :", round(recall_score(y_test, pred), 4))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, "model.joblib", compress=3)
print("Saved model.joblib")
