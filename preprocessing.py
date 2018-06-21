import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
PCA_COMPONENT_AMOUNT = 200
TRAINING_SET_FILE = "train_no_constant_scaled_PCA.pkl"


def remove_constant_columns(df):  # Removes columns with 0 variance
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)


def separate_labels(df):
    targets = df.pop("target")  # Training set now contains only variables
    return targets


def separate_id(df):
    ids = df.pop("ID")  # Removing String ID for feature scaling
    return ids


def apply_pca(df, features=200):  # Applies PCA on a dataframe and returns another with labeled features
    scaled_df = StandardScaler().fit_transform(df)
    pca = PCA(n_components=features)
    pca_df = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame(pca_df, columns=[i for i in range(features)])
    return pca_df


for file in ["./raw_files/test.csv", "./raw_files/train.csv"]:

    dataframe = pd.read_csv(file)
    IDs = separate_id(dataframe)

    if "train" in file:
        labels = separate_labels(dataframe)
        labels.to_pickle("trainY.pkl")
    else:
        IDs.to_pickle("testID.pkl")

    pca_dataframe = apply_pca(dataframe, PCA_COMPONENT_AMOUNT)
    pca_dataframe.to_pickle("testX.pkl" if "test" in file else "trainX.pkl")

""" 
IDs are unique
training_set = pd.read_pickle(TRAINING_SET_FILE)
print(len(training_set["ID"].unique()))
print(len(training_set["ID"]))
"""
"""
pc_training_set["ID"] = IDs
pc_training_set["targets"] = targets
print(pc_training_set.head())
pc_training_set.to_pickle("train_no_constant_scaled_PCA.pkl")
"""
