import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

trainX = pd.read_pickle("trainX.pkl")
trainY = pd.read_pickle("trainY.pkl")
testX = pd.read_pickle("testX.pkl")
testIDs = pd.read_pickle("testID.pkl")

model = RandomForestRegressor(n_jobs=-1)

model.set_params(n_estimators=100)
model.fit(trainX, trainY)
resultframe = pd.DataFrame()
resultframe["ID"] = testIDs

resultframe["target"] = pd.DataFrame(model.predict(testX), dtype="float", columns=["target"])
print(resultframe.head())
resultframe.to_csv("predictions.csv", index=False)
