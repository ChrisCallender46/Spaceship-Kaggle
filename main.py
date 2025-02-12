import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

passengers = pd.read_csv("datasets/spaceship_data/train.csv")
test = pd.read_csv("datasets/spaceship_data/test.csv")
test_ids = test["PassengerId"]

def clean(data):
    data["cabin_letter"] = data.Cabin.apply(lambda x: str(x)[0])
    data["money_spent"] = data["RoomService"] + data["FoodCourt"] + data["ShoppingMall"] + data["Spa"] + data["VRDeck"]

    data = data.drop(["Cabin", "Name", "PassengerId"], axis=1)

    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "money_spent"]
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Replace null values with mode
    data["HomePlanet"].fillna("Earth", inplace=True)
    data["CryoSleep"].fillna(False, inplace=True)
    data["Destination"].fillna("TRAPPIST-1e", inplace=True)
    data["VIP"].fillna(False, inplace=True)

    return data

passengers = clean(passengers)
test = clean(test)

le = preprocessing.LabelEncoder()

cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "cabin_letter"]

for col in cols:
    passengers[col] = le.fit_transform(passengers[col])
    test[col] = le.transform(test[col])
    print(le.classes_)

y = passengers["Transported"]
X = passengers.drop(["Transported"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=passengers["Transported"], random_state=42)

model = GradientBoostingClassifier(random_state=42)
model = model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))

submission_preds = model.predict(test)

df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Transported": submission_preds})

df.to_csv("spaceship_submission.csv", index=False)
