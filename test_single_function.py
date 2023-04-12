   

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import support_function as sf
plt.close('all') 


feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
cnt=0
# plt.figure()
sns.set() 
fig, axes = plt.subplots(3, 6, sharey=True)
fig.suptitle('Class features composition')
for feature in feature_name:
    sub_df=df.groupby(["predict", feature])[feature].count().reset_index(name="count")
    sub_df['perc'] = sub_df['count'].groupby(sub_df['predict']).transform(lambda x: x/x.sum())
    sub_df['predict']=sub_df['predict'].map(species_dict)     
    sns.barplot(x ="predict", y = "perc", data = sub_df, hue = feature, ax=axes[cnt//len(feature_name),cnt%len(feature_name)])    
    axes[row,col].set_title(feature)    
    cnt=cnt+1


feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
cnt=0
# plt.figure()
sns.set() 
fig, axes = plt.subplots(3, 6, sharey=True)
fig.suptitle('Class features composition')
for feature in feature_name:
    sub_df=df.groupby(["predict", feature])[feature].count().reset_index(name="count")
    sub_df['perc'] = sub_df['count'].groupby(sub_df['predict']).transform(lambda x: x/x.sum())
    sub_df['predict']=sub_df['predict'].map(species_dict)     
    sns.barplot(x ="predict", y = "perc", data = sub_df, hue = feature, ax=axes[cnt//len(feature_name),cnt%len(feature_name)])    
    axes[row,col].set_title(feature)    
    cnt=cnt+1    
    
    
