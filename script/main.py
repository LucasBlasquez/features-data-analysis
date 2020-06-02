import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

filepath = "../res/high_diamond_ranked_10min.csv"
lol_data = pd.read_csv(filepath, index_col="gameId")
lol_data.head()

#Apply in lol_data blueWins replace 1 'Yes' and 0 'No' 
def new_wins(wins):
    if wins.blueWins == 1:
        wins.blueWins = "yes"
        return wins
    else:
        wins.blueWins = "no"
        return wins

new_lol_data = lol_data.apply(new_wins,axis='columns')
new_lol_data.head(10)

#Distribuiton chart of blue team victories x blue gold per minute
sns.swarmplot(x=new_lol_data['blueWins'],y=new_lol_data['blueGoldPerMin'])
plt.title("Relation of Gold per Minute and Victorys in the first 10 min of High Elo Ranked Games of LoL")
plt.ylabel("Gold Per Minute")
plt.xlabel("Victory")

#features and target of (only) blue team to Machine Learning
X = lol_data.iloc[:,1:20]
y = lol_data['blueWins']

#test_size is default 0.25, that is 25% of data to train and 75% to test
train_X, val_X, train_y, val_y = train_test_split(X, y)

#Creating the Machine Learning model with ExtraTreesClassifier class
model = ExtraTreesClassifier()
model.fit(train_X,train_y)

#accuracy of model predict is ~73%
result = model.score(val_X, val_y)
predict = model.predict(train_X)