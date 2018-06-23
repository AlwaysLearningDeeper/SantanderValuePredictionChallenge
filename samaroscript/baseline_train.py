import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

trainX = pd.read_pickle("trainX.pkl")
trainY = pd.read_pickle("trainY.pkl")
testX = pd.read_pickle("testX.pkl")
testIDs = pd.read_pickle("testID.pkl")

trainX, trainXtest, trainY, trainYtest = train_test_split(trainX, trainY, test_size=0.2, random_state=7)

# Random forest regressor does not support custom loss functions like the one being used to evaluate in kaggle
model = RandomForestRegressor(n_jobs=-1)

scores = []
estimators = np.arange(40, 500, 10)
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(trainX, trainY)
    scores.append(model.score(trainXtest, trainYtest))

plt.plot(estimators, scores)
plt.show()

model.set_params(n_estimators=scores.index(max(scores))*10 + 10)
model.fit(trainX, trainY)
resultframe = pd.DataFrame()
resultframe["ID"] = testIDs

resultframe["target"] = pd.DataFrame(model.predict(testX), dtype="float", columns=["target"])
print(resultframe.head())
resultframe.to_csv("predictions.csv", index=False)
