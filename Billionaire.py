import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/Billionaire.csv")
#data.to_csv('Billionaire.csv')
data=pd.read_csv('Billionaire.csv')
print(data.head())
data.info()
data.Age.isnull().sum()
print(data.isnull().sum())
data=data.dropna()
data.NetWorth=data.NetWorth.str.strip('$')
data.NetWorth=data.NetWorth.str.strip('B')
data.NetWorth=data.NetWorth.astype('float')
df=data.sort_values(by='NetWorth',ascending=False).head(10)
print(df[['Name','NetWorth']])
plt.figure(figsize=(20, 10))
sns.histplot(x="Name", hue="NetWorth", data=df)
plt.show()
a=data['Source'].value_counts().head()
index=a.index
Source=a.values
custom_colors = ["skyblue", "yellowgreen", 'tomato', "blue", "red"]
plt.figure(figsize=(5, 5))
plt.pie(Source, labels=index, colors=custom_colors)

central_circle = plt.Circle((0, 0),0.2, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
#plt.rc('font', size=12)
plt.title("Top 5 Domains to Become a Billionaire", fontsize=20)
plt.show()