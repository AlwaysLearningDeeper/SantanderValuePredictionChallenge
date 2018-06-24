import pandas as pd
import seaborn as sns

train = pd.read_csv('../raw_files/train.csv')
test = pd.read_csv('../raw_files/test.csv')

corr = train.corr()
heatmap = sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values)
fig = heatmap.get_figure()
fig.savefig('figure.png')