from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# same split as the eval in train.py
X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0)

models = {
    "K-nearest neighbors": KNeighborsClassifier(),
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
