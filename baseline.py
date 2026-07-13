from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# same dataset and split as the eval in train.py
# phoneme (OpenML 1489): 5404 rows x 5 numeric features, binary, nonlinear
X, y = fetch_openml(data_id=1489, return_X_y=True, as_frame=False, parser="liac-arff")
y = (y == "2").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

models = {
    "Decision tree": DecisionTreeClassifier(random_state=0),
    "Random forest": RandomForestClassifier(random_state=0),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
}

if __name__ == "__main__":
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        print(f"{name:22s} roc_auc {roc_auc_score(y_test, prob):.4f}  acc {accuracy_score(y_test, pred):.4f}")
