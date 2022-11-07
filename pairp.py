import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("dark")
df = pd.read_csv('Advertising.csv')
#get a snippet of our data to look at data formats and size
#df.head()
print(df)
#make pairplots
# sns.pairplot(df, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)
sns.pairplot(df, x_vars=['TV','Radio','Newspaper','Sales'], y_vars=['TV','Radio','Newspaper','Sales'],kind='scatter')
plt.show()