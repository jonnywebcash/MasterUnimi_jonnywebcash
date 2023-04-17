import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import (mutual_info_score, normalized_mutual_info_score,adjusted_mutual_info_score)
from sklearn.metrics import (homogeneity_score,completeness_score,v_measure_score)

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
    # Creazione matrice correlazione
    corr_matrix = df_tmp.corr().abs()
    # grafico della heatmap
    sns.heatmap(corr_matrix)
    
    # Selezione del tringolo superiore della matrice di correlazione
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Estraiamo tutte le combinazioni di feature e le filtriamo per avere solo le correlazioni tra i type e le feature
    s = upper.unstack()
    added_dummy_cols = [item for item in list(df_tmp.columns.values) if (item not in(df.columns.values))]
    so=s[s.index.isin(added_dummy_cols, level=0)]
    so=so[~so.index.isin(added_dummy_cols, level=1)]
    so = so.sort_values(kind="quicksort",ascending=False, na_position='last')
    so.columns =['corr']
    so.index=so.index.set_names(['type','feature'])
    so=so.reset_index()
    so.columns=['type','feature','corr']
    
    plt.figure()
    cmap =plt.colormaps['jet']
    
    cnt=1
    for cluster in added_dummy_cols:
        sub_df=so[so["type"]==cluster]
        sub_df=sub_df.sort_values('feature')
        
        plt.subplot(2, 4,cnt)
        plt.title(cluster)
        plt.bar(sub_df['feature'], sub_df['corr'], align='center',color=cmap(sub_df['corr']))     
        #plt.xlabel(sub_df['feature'].values.tolist())
        plt.xticks(rotation = 60)
        plt.ylabel("corr")    
        plt.ylim(ymin = 0,ymax = 1)
        plt.grid(True)
        #plt.tight_layout()
        cnt=cnt+1
    # Show the plots
    plt.show()

    




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
        
# Assegna una label al cluster in base alla classe di maggioranza        
def majority_voting_label(df):
    
    predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
    predict_major=df.set_index('predict').to_dict()['type']
    df['predict_label']=df['predict'].map(predict_major)
    return(df)


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
    df['predict_label']=df['predict_label'].map(species_dict)  
    df['type']=df['type'].map(species_dict)    
    return(df)


def scatter_plot_result(df):
    df['type'] = df['type'].apply(lambda x: x + np.random.randint(-40,40)/100)
    df['predict_label'] = df['predict_label'].apply(lambda x: x + np.random.randint(-40,40)/100)
    groups = df.groupby('predict')
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
    #plt.scatter(df['type'], df['predict_label'], c=df['predict'], cmap='viridis')
    for cluster, group in groups:
        plt.plot(group.type, group.predict_label, marker='o', linestyle='', markersize=8, label=cluster)
    
    plt.xlabel('type')
    plt.ylabel('majority voting label')
    ax.add_patch(circle0)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    ax.add_patch(circle5)
    ax.add_patch(circle6)
    plt.grid()
    plt.title(df['model_name'][0])
    plt.legend()
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

def barplot_class_feature_comparison(df,species_dict, major_label_voting):
   if major_label_voting==1:
        predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
        predict_major=df.set_index('predict').to_dict()['type']
        for k in predict_major:
            predict_major[k]=str(k)+'_'+species_dict[predict_major[k]]
        
   feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
   cnt=0
   sns.set() 
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
       sub_df=df.groupby(["predict", feature])[feature].count().reset_index(name="count")
       sub_df['perc'] = sub_df['count'].groupby(sub_df['predict']).transform(lambda x: x/x.sum())
       if major_label_voting==1:           
           sub_df['predict']=sub_df['predict'].map(predict_major)
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
        if major_label_voting==1:           
            sub_df['predict']=sub_df['predict'].map(predict_major)
        sns.barplot(x ="predict", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
        axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)        
        # axes[cnt//4,cnt%4].set_title(feature)        
        cnt=cnt+1

def classification_percentage_performance(df,species_dict):
    df=replace_predict_with_major(df).copy()    
    sub_df=df.groupby(["type", "predict_label"])["predict"].count().reset_index(name="count")
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

def rand_index(df):
    ri = rand_score(df['type'], df['predict'])
    return(ri)

def adjusted_rand_index(df):
    ari = adjusted_rand_score(df['type'], df['predict'])
    return(ari)

def mutual_information_index(df):
    MI = mutual_info_score(df['type'], df['predict'])
    NMI = normalized_mutual_info_score(df['type'], df['predict'])
    AMI = adjusted_mutual_info_score(df['type'], df['predict'])
    return(MI,NMI,AMI)

def v_measure_index(df):
    HS = homogeneity_score(df['type'], df['predict'])
    CS = completeness_score(df['type'], df['predict'])
    V = v_measure_score(df['type'], df['predict'], beta=1.0)
    return(HS,CS,V)

def silhouette_score_index(df):
    ss = silhouette_score(df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']], df['predict'])
    return(ss)

def model_performance_evaluation(df_full,df,model_comparison):
    
     # Similarity index
    #similarity_score=similarity_index(df)
    # Dissimilarità index
    #dissimilarity_score=dissimilarity_index(df)
    
    #Rand Index
    # Misura la somiglianza tra le assegnazioni del cluster effettuando confronti a coppie. Un punteggio più alto indica una somiglianza maggiore.
    # RI = (numero di previsioni corrette a coppie) / (numero totale di possibili coppie)
    ri_score=rand_index(df)
    
    # Adjusted Rand Index
    #L'indice RAND regolato è una metrica di valutazione che viene utilizzata per misurare la
    # somiglianza tra due clustering considerando tutte le coppie di N_SAMPLE e calcolando le coppie
    # di conteggio delle stesse o diversi cluster nel raggruppamento effettivo e previsto.
    #ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    # A score above 0.7 is considered to be a good match. 
    ari_score=adjusted_rand_index(df)
    
    # Mutual Information (MI, NMI, AMI)
    # Le informazioni reciproche (MI, NMI, AMI) misurano l'accordo tra le assegnazioni del cluster.
    # Un punteggio più alto indica una somiglianza maggiore.
    MI_score,NMI_score,AMI_score=mutual_information_index(df)
    
    # V-measure
    # V-Measure misura la correttezza delle assegnazioni del cluster usando l'analisi dell'entropia condizionale.
    # Un punteggio più alto indica una somiglianza maggiore
    # Omogeneità: ogni cluster contiene solo membri di una singola classe (un po 'come "precisione")
    # Completezza: tutti i membri di una determinata classe sono assegnati allo stesso cluster (un po 'come "richiamo")
    HS_score,CS_score,V_score=v_measure_index(df)
    
    # Silhouette Score aka Silhouette Coefficient
    # Silhouette score aka Silhouette Coefficient is an evaluation metric that results in the range of -1 to 1. A score near 1 signifies the best importance that the data point is very compact within the cluster to which it belongs and far away from the other clusters. The score near -1 signifies the least or worst importance of the data point. A score near 0 signifies overlapping clusters. 
    ss_score=silhouette_score_index(df_full)
    
    new_row=[[df['model_name'][0],ri_score,ari_score,MI_score,NMI_score,AMI_score,HS_score,CS_score,V_score,ss_score]]
    
    model_comparison= model_comparison.append(pd.DataFrame(new_row, columns=model_comparison.columns))
    return(model_comparison)




   

def kmeans_prediction(df, k):
    # Definiamo il numero K di cluster presenti nei dati:
    kmeans = KMeans(n_clusters=k)
    # Addestra il modello di K-means sul dataset
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    kmeans.fit(features)
    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = kmeans.predict(features)
    df['model_name'] = 'kmeans_'+str(k)
    majority_voting_label(df)
    return(df)

def dbscan_prediction(df,eps_v,min_sam):
    dbscan_model = DBSCAN( eps = eps_v, min_samples = min_sam)
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    dbscan_model.fit(features)    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = dbscan_model.labels_
    df['model_name'] = 'dbscan_eps_'+str(eps_v)+'_min_samples_'+str(min_sam)
    majority_voting_label(df)
    return(df)

