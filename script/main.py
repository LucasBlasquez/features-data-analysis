import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

filepath = "../res/high_diamond_ranked_10min.csv"
lol_data = pd.read_csv(filepath)

#Apply in lol_data blueWins replace 1 'Yes' and 0 'No' 
def new_wins(wins):
    if wins.blueWins == 1:
        wins.blueWins = "yes"
        return wins
    else:
        wins.blueWins = "no"
        return wins

new_lol_data = lol_data.apply(new_wins,axis='columns')

#Distribuiton chart of blue team victories x blue gold per minute
sns.swarmplot(x=new_lol_data['blueWins'],y=new_lol_data['blueGoldPerMin'])
plt.title("Relation of Gold per Minute and Victorys in the first 10 min of High Elo Ranked Games of LoL")
plt.ylabel("Gold Per Minute")
plt.xlabel("Victory")

#Converting categorical variables into numerical features to improve the model
features = lol_data.iloc[:,2:20]
encoder = LabelEncoder()
feat_lol = features.apply(encoder.fit_transform)

#features and target of (only) blue team to Machine Learning
X = feat_lol
y = lol_data['blueWins']

#test_size is default 0.25, that is 25% of data to train and 75% to test
train_X, val_X, train_y, val_y = train_test_split(X, y)

#Creating the Machine Learning model with ExtraTreesClassifier class
model = ExtraTreesClassifier()
model.fit(train_X,train_y)

#score/accuracy of model predict is ~73%
result = model.score(val_X, val_y)
predict = model.predict(train_X)