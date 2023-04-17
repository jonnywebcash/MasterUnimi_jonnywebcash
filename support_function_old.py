import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.close('all') 

# Generazione della heatmap valutando la correlazione delle singole features con le singole specie
def feature_vs_class_heatmap(df,species_dict):
    df_tmp=df.copy()
    df_tmp['type']=df_tmp['type'].map(species_dict)    
    # One-hot encoding
    df_tmp=pd.get_dummies(df_tmp,columns=["type"])
    sns.heatmap(df_tmp.corr())

# Barplot della distribuzione delle varie classi all'interno del dataset
def class_distributuion_barplot(df,species_dict):
    sub_df=df.groupby(["type"])["type"].count().reset_index(name="count").copy()
    sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/df.shape[0])
    sub_df['type']=sub_df['type'].map(species_dict)
    sns.set() 
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Dataset composition')
    sns.barplot(x ="type", y = 'perc', data = sub_df, hue = "type",ax=axes[0])    
    sns.barplot(x ="type", y = 'count', data = sub_df, hue = "type",ax=axes[1])

# Barplot della distribuzione percentuale all'interno di ogni classe dei valori delle singole features 
def barplot_class_feature_distribution(df,species_dict):
   feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
   cnt=0
   sns.set() 
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
       sub_df=df.groupby(["type", feature])[feature].count().reset_index(name="count").copy()
       sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
       sub_df['type']=sub_df['type'].map(species_dict) 
       sns.barplot(x ="type", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
       axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)
       # axes[cnt//4,cnt%4].set_title(feature)       
       cnt=cnt+1
    
   feature_name=['backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
   cnt=0
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
        sub_df=df.groupby(["type", feature])[feature].count().reset_index(name="count").copy()
        sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
        sub_df['type']=sub_df['type'].map(species_dict) 
        sns.barplot(x ="type", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
        axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)        
        # axes[cnt//4,cnt%4].set_title(feature)        
        cnt=cnt+1


def replace_predict_with_major(df):
    # Valutazione predict
    #predict_major={}
    # for result_predict in (df['predict'].unique()):
    #     sub_df=df[df["predict"]==result_predict]
    #     major_species=sub_df['type'].value_counts().idxmax()
    #     predict_major[int(result_predict)]=major_species
    predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
    predict_major=df.set_index('predict').to_dict()['type']
    # for chiave in predict_major.keys():
    #     df['predict'] = df['predict'].replace([int(chiave)], species_dict[predict_major[chiave]])
    df['predict']=df['predict'].map(predict_major)
    return(df)

 
def replace_species(df,species_dict):
    df=replace_predict_with_major(df).copy()
    df['predict']=df['predict'].map(species_dict)  
    df['type']=df['type'].map(species_dict)    
    # for chiave in species_dict.keys():
    #     df['type'] = df['type'].replace([int(chiave)], species_dict[chiave])
        #sub_df_frequency = sub_df['type'].value_counts()
    return(df)



def scatter_plot_result(df):
    # df_tmp = pd.DataFrame(columns = list(df.columns.values))
    # # Valutazione predict
    # for result_predict in (df['predict'].unique()):
    #     sub_df=df[df["predict"]==result_predict]
    #     major_species=sub_df['type'].value_counts().idxmax()
    #     sub_df['predict'] = sub_df['predict'].replace(result_predict, major_species)
    #     df_tmp=pd.concat([df_tmp, sub_df], axis=0)

    # df=df_tmp
    df=replace_predict_with_major(df).copy()
    df['type'] = df['type'].apply(lambda x: x + np.random.randint(-40,40)/100)
    df['predict'] = df['predict'].apply(lambda x: x + np.random.randint(-40,40)/100)
    # Crea un grafico a dispersione dei dati utilizzando le colonne Età e Voto
    fig = plt.figure()
    ax = fig.gca()
    circle0 = plt.Circle((0,0), 0.5,  fill=False)
    circle1 = plt.Circle((1,1), 0.5,  fill=False)
    circle2 = plt.Circle((2,2), 0.5,  fill=False)
    circle3 = plt.Circle((3,3), 0.5,  fill=False)
    circle4 = plt.Circle((4,4), 0.5,  fill=False)
    circle5 = plt.Circle((5,5), 0.5,  fill=False)
    circle6 = plt.Circle((6,6), 0.5,  fill=False)

    ax.set_xticks(np.arange(-1, 7, 1))
    ax.set_yticks(np.arange(-1, 7, 1))
    plt.scatter(df['type'], df['predict'], c=df['type'], cmap='viridis')
    plt.xlabel('type')
    plt.ylabel('predict')
    ax.add_patch(circle0)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.add_patch(circle5)
    ax.add_patch(circle6)
    plt.grid()
    plt.show()
    return(df)



def barplot_class_feature_distribution(df,species_dict):
   feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
   cnt=0
   sns.set() 
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
       sub_df=df.groupby(["type", feature])[feature].count().reset_index(name="count").copy()
       sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
       sub_df['type']=sub_df['type'].map(species_dict) 
       sns.barplot(x ="type", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
       axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)
       # axes[cnt//4,cnt%4].set_title(feature)       
       cnt=cnt+1
    
   feature_name=['backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
   cnt=0
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
        sub_df=df.groupby(["type", feature])[feature].count().reset_index(name="count").copy()
        sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
        sub_df['type']=sub_df['type'].map(species_dict) 
        sns.barplot(x ="type", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
        axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)        
        # axes[cnt//4,cnt%4].set_title(feature)        
        cnt=cnt+1

def barplot_class_feature_percentage(df,species_dict):
    feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','type']
    for result_predict in (df['predict'].unique()):
        
        sub_df=df[df["predict"]==result_predict].copy() 
        cnt=1
        plt.figure()
        for feature in feature_name:
            # get colum value distribution
            a = (sub_df[feature].value_counts(normalize=True) 
                        .mul(100)
                        .rename_axis(feature)
                        .reset_index(name='percentage'))
            # creating the bar plot
            
            plt.subplot(3, 6,cnt)
            #plt.title(feature)
            plt.bar(a[feature].values.tolist(), a['percentage'], align='center', color ='b')
         
            plt.xlabel(feature)
            plt.ylabel("Percentage [%]")    
            #ax[cnt].plt.show()
            cnt=cnt+1
        # Show the plots
        sub_df['predict']=sub_df['predict'].map(species_dict) 
        plt.suptitle("predict:"+sub_df['predict'].unique())
        plt.show()

def barplot_class_feature_comaprison(df,species_dict):
   df=replace_predict_with_major(df)
   feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
   cnt=0
   sns.set() 
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
       sub_df=df.groupby(["predict", feature])[feature].count().reset_index(name="count")
       sub_df['perc'] = sub_df['count'].groupby(sub_df['predict']).transform(lambda x: x/x.sum())
       sub_df['predict']=sub_df['predict'].map(species_dict) 
       sns.barplot(x ="predict", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
       axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)
       # axes[cnt//4,cnt%4].set_title(feature)       
       cnt=cnt+1
    
   feature_name=['backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
   cnt=0
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
        sub_df=df.groupby(["predict", feature])[feature].count().reset_index(name="count")
        sub_df['perc'] = sub_df['count'].groupby(sub_df['predict']).transform(lambda x: x/x.sum())
        sub_df['predict']=sub_df['predict'].map(species_dict) 
        sns.barplot(x ="predict", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
        axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)        
        # axes[cnt//4,cnt%4].set_title(feature)        
        cnt=cnt+1

def classification_percentage_performance(df,species_dict):
    df=replace_predict_with_major(df).copy()    
    sub_df=df.groupby(["type", "predict"])["predict"].count().reset_index(name="count")
    sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
    if   type(sub_df['predict'][0]) == int or type(sub_df['predict'][0]) == float:
        sub_df['predict']=sub_df['predict'].map(species_dict)
    if   type(sub_df['type'][0]) == int or type(sub_df['type'][0]) == float:
        sub_df['type']=sub_df['type'].map(species_dict)
    sns.set() 
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Distribution of classified classes')
    sns.barplot(x ="type", y = 'perc', data = sub_df, hue = "predict",ax=axes[0])    
    sns.barplot(x ="type", y = 'count', data = sub_df, hue = "predict",ax=axes[1])
    

def similarity_index(df):
    # df=df_tmp
    predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
    predict_major=df.set_index('predict').to_dict()['type']
    if   type(df['predict'][0]) == int or type(df['predict'][0]) == float:
        df['predict']=df['predict'].map(predict_major)
    

    df=replace_predict_with_major(df)
    similarity_dict={}
    for real_type in (df['type'].unique()):
        sub_df=df[df["type"]==real_type]
        match=sub_df[sub_df["predict"]==real_type]
        if match.shape[0]>0:
            similarity_dict[real_type]=match.shape[0]/sub_df.shape[0]*100
        else:
            similarity_dict[real_type]=0
    myKeys = list(similarity_dict.keys())
    myKeys.sort()
    sorted_dict = {i: similarity_dict[i] for i in myKeys}
    similarity_dict=sorted_dict
    
    fig = plt.figure()
    ax = fig.gca()
    plt.bar(range(len(similarity_dict)), list(similarity_dict.values()), tick_label=list(similarity_dict.keys()))
    plt.xlabel('type')
    plt.ylabel('Similarity inside the group')
    plt.show()
    return(similarity_dict)

def dissimilarity_index(df):
    df_tmp = pd.DataFrame(columns = list(df.columns.values)).copy()
    # Valutazione predict
    for result_predict in (df['predict'].unique()):
        sub_df=df[df["predict"]==result_predict]
        major_species=sub_df['type'].value_counts().idxmax()
        sub_df['predict label'] =major_species
        df_tmp=pd.concat([df_tmp, sub_df], axis=0)

    df=df_tmp
    dissimilarity_dict={}
    for real_type in (df['type'].unique()):
        sub_df=df[df["type"]==real_type]
        match=sub_df[sub_df["predict label"]==real_type]
        dissimilarity_dict[real_type]=((sub_df.shape[0]-match.shape[0])/sub_df.shape[0])*100
    myKeys = list(dissimilarity_dict.keys())
    myKeys.sort()
    sorted_dict = {i: dissimilarity_dict[i] for i in myKeys}
    dissimilarity_dict=sorted_dict
    
    fig = plt.figure()
    ax = fig.gca()
    plt.bar(range(len(dissimilarity_dict)), list(dissimilarity_dict.values()), tick_label=list(dissimilarity_dict.keys()))
    plt.xlabel('type')
    plt.ylabel('Dissimilarity inside the group')
    plt.show()
    return(dissimilarity_dict)

def adjusted_rand_index(df):
    ari = adjusted_rand_score(df['type'], df['predict'])
    return(ari)

def rand_index(df):
    ri = rand_score(df['type'], df['predict'])
    return(ri)

def silhouette_score_index(df):
    ss = silhouette_score(df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']], df['predict'])
    return(ss)

   

def kmeans_prediction(df, cluster_num):
    # Definiamo il numero K di cluster presenti nei dati:
    # Crea un'istanza di KMeans con 7 cluster
    kmeans = KMeans(n_clusters=cluster_num)
    # Addestra il modello di K-means sul dataset
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    kmeans.fit(features)
    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = kmeans.predict(features)
    return(df)

def dbscan_prediction(df):
    dbscan_model = DBSCAN( eps = 0.75, min_samples = 5)
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']]
    dbscan_model.fit(features)

