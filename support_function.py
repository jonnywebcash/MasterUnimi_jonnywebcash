import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close('all') 

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
    df=replace_predict_with_major(df)
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
    df=replace_predict_with_major(df)
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

def similarity_index(df):
    # df_tmp = pd.DataFrame(columns = list(df.columns.values)+['predict label'])
    # # Valutazione predict
    # for result_predict in (df['predict'].unique()):
    #     sub_df=df[df["predict"]==result_predict]
    #     major_species=sub_df['type'].value_counts().idxmax()
    #     sub_df['predict label'] =major_species
    #     df_tmp=pd.concat([df_tmp, sub_df], axis=0)

    # df=df_tmp
    predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
    predict_major=df.set_index('predict').to_dict()['type']
    df['predict label']=df['predict'].map(predict_major)

    df=replace_predict_with_major(df)
    similarity_dict={}
    for real_type in (df['type'].unique()):
        sub_df=df[df["type"]==real_type]
        match=sub_df[sub_df["predict label"]==real_type]
        similarity_dict[real_type]=match.shape[0]/sub_df.shape[0]*100
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
    df_tmp = pd.DataFrame(columns = list(df.columns.values)+['predict label'])
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

def feature_comon_group(df):
        # plot a Stacked Bar Chart using matplotlib
    df.plot(
      x = 'type', 
      kind = 'barh', 
      stacked = True, 
      title = 'Percentage Stacked Bar Graph', 
      mark_right = True)
      
    df_total = df["Studied"] + df["Slept"] + df["Other"]
    df_rel = df[df.columns[1:]].div(df_total, 0)*100
      
    for n in df_rel:
        for i, (cs, ab, pc) in enumerate(zip(df.iloc[:, 1:].cumsum(1)[n], 
                                             df[n], df_rel[n])):
            plt.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%', 
                     va = 'center', ha = 'center')
    fig, ax = plt.subplots()
    bottom = np.zeros(7)
    
    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    
    ax.set_title("Number of penguins with above average body mass")
    ax.legend(loc="upper right")
    
    plt.show()
    

def kmeans_prediction(df):
    # Definiamo il numero K di cluster presenti nei dati:
    # Crea un'istanza di KMeans con 7 cluster
    kmeans = KMeans(n_clusters=7)
    # Addestra il modello di K-means sul dataset
    kmeans.fit(df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']])
    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = kmeans.predict(df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']])
    return(df)

