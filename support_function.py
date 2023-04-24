import sklearn
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import (mutual_info_score, normalized_mutual_info_score,adjusted_mutual_info_score)
from sklearn.metrics import (homogeneity_score,completeness_score,v_measure_score)
from sklearn.metrics import confusion_matrix

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
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
    
    # Barplot delle correlazioni vs le features
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

# Generazione della heatmap valutando la correlazione delle singole features con i singoli cluster
def feature_vsclustered_class(df,df2,species_dict):
    # Original dataset
    df_tmp=df.copy()
    df_tmp['type']=df_tmp['type'].map(species_dict)    
    # One-hot encoding
    df_tmp=pd.get_dummies(df_tmp,columns=["type"])
    # Creazione matrice correlazione
    corr_matrix = df_tmp.corr().abs()
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
    
    ## Mapped dataset
    df_tmp2=df2[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    df_tmp2['predict_label']=df2['predict_label'].map(species_dict)    
    df_tmp2['cluster_']=df_tmp2['predict_label']+'-'+df2['predict'].astype(str)
    # One-hot encoding
    df_tmp2=pd.get_dummies(df_tmp2,columns=["cluster_"])
    # Creazione matrice correlazione
    corr_matrix2 = df_tmp2.corr().abs()    
    # Selezione del tringolo superiore della matrice di correlazione
    upper2 = corr_matrix2.where(np.triu(np.ones(corr_matrix2.shape), k=1).astype(bool))    
    # Estraiamo tutte le combinazioni di feature e le filtriamo per avere solo le correlazioni tra i type e le feature
    s2 = upper2.unstack()
    added_dummy_cols2 = [item for item in list(df_tmp2.columns.values) if (item not in(df2.columns.values))]
    so2=s2[s2.index.isin(added_dummy_cols2, level=0)]
    so2=so2[~so2.index.isin(added_dummy_cols2, level=1)]
    so2 = so2.sort_values(kind="quicksort",ascending=False, na_position='last')
    so2.columns =['corr']
    so2.index=so2.index.set_names(['cluster','feature'])
    so2=so2.reset_index()
    so2.columns=['cluster','feature','corr']
    
    cmap =plt.colormaps['jet']
    cnt=1;
    plt.figure()
    for cluster in added_dummy_cols2:
        if cnt>2:
            plt.show()
            cnt=1;
            plt.figure()
            
        sub_df2=so2[so2["cluster"]==cluster]
        sub_df2=sub_df2.sort_values('feature')
        
        label_majority=cluster.split('-')
        label_majority=label_majority[0].replace("cluster__", "")
        sub_df=so[so["type"]=='type_'+label_majority]
        sub_df=sub_df.sort_values('feature')
        
        plt.subplot(2,2,cnt)
        plt.title(cluster)        
        plt.bar(sub_df2['feature'], sub_df2['corr'], align='center',color=cmap(sub_df2['corr']),width=0.4)     
        plt.xticks(rotation = 60)
        plt.ylabel("Cluster feature corr")  
        plt.ylim(ymin = 0,ymax = 1)
        plt.grid(True)
 
        plt.subplot(2, 2,cnt+2)
        plt.title(label_majority)        
        plt.bar(sub_df['feature'], sub_df['corr'], align='center',color=cmap(sub_df['corr']),width=0.4)     
        #plt.xlabel(sub_df['feature'].values.tolist())
        plt.xticks(rotation = 60)
        plt.ylabel("Majority Specie feature corr")    
        plt.ylim(ymin = 0,ymax = 1)
        plt.grid(True)
        
        cnt=cnt+1
    plt.show()
       
       

    




# Barplot della distribuzione delle varie classi all'interno del dataset
def class_distributuion_barplot(df,species_dict):
    # Ragruppiamo il dataset per "specie" e tramite count() e lambda function computiamo le occorrenze e le percentuali per ogni specie
    sub_df=df.groupby(["type"])["type"].count().reset_index(name="count").copy()
    sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/df.shape[0])
    sub_df['type']=sub_df['type'].map(species_dict)
    sns.set() 
    # Barplot di occorenze e percentuale dei campioni del dataset per ogni specie
    fig, axes = plt.subplots(2, 1)
    fig.suptitle('Dataset composition')
    sns.barplot(x ="type", y = 'perc', data = sub_df, hue = "type",ax=axes[0])    
    sns.barplot(x ="type", y = 'count', data = sub_df, hue = "type",ax=axes[1])

# Barplot della distribuzione percentuale all'interno di ogni classe dei valori delle singole features 
def barplot_class_feature_distribution(df,species_dict):
# Splittiamo in due la lista delle feature così da poter plottare i bar plot delle distribuzioni delle singole features
# per le singole specie in due immagini composte da 8 bar plot (griglia 2x4)
   feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
   cnt=0
   sns.set() 
   fig, axes = plt.subplots(2, 4, sharey=True)
   fig.suptitle('Class features composition')
   for feature in feature_name:
       # raggruppiamo il dataset combinazione di feature e specie, così da poter calcolare i conteggi e poi le percentuali di
       # distribuzione dei valori di ogni singola feature per ogni specie
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
    # raggruppiamo il datatset per predizione, all'interno di ogni gruppo contiamo la specie rappresentata maggiormente, essa sarà
    # la label assegnata tramite Majority Voting al singolo cluster
    predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
    # Trasformiamo in dizionario la relazione predizione-label MV per poi poterne mappare i valori nel dataframe
    predict_major=pd.Series(predict_major.type.values,index=predict_major.predict).to_dict()
    # Assugna la label ad una nuova colonna del dataframe 'predict_label'
    df['predict_label']=df['predict'].map(predict_major)
    return(df)


def replace_predict_with_major(df):
    # raggruppiamo il datatset per predizione, all'interno di ogni gruppo contiamo la specie rappresentata maggiormente, essa sarà
    # la label assegnata tramite Majority Voting al singolo cluster
    predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
    predict_major=df.set_index('predict').to_dict()['type']
    # Assugna la label alla colonna 'predict' del dataframe
    df['predict']=df['predict'].map(predict_major)
    return(df)

# funzione per la sostituzione del valore numerico presente in  'predict_label' con il nome della specie
# presente nel dizionarion "species_dict"
def replace_species(df,species_dict):
    df['predict_label']=df['predict_label'].map(species_dict)  
    df['type']=df['type'].map(species_dict)    
    return(df)

# Plot della matrice di confusione
def confusion_matrix_plot(df):
    # tramite la funzione "confusion_matrix" computa la matrice di confusione tra specie originali e label (MV)
    # assegnata al ogni elemento del dataset clusterizzato 
    cm = confusion_matrix(df['type'],df['predict_label'])
    sns.set() 
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted Class by majority voting")
    plt.ylabel("True Class")
    plt.title('Confusion matrix:'+df['model_name'][0])
    plt.show()

# Esegue lo scatter plot della specie originale (asse x) vs specie assegnata per MV (asse y)
# assegnando un colore diverso per ogni cluster di appartenenza
# Ci permette in un grafico solo di:
#    - vedere in quanti cluster è stato suddivisomil dataset originale
#    - osservare visivamente l'omogeneità rispetto alla specie originale di ogni cluster
#    - osservare tramite il metodo del MV quanti "errate clusterizzazioni" e elevate suddivisioni dentro la stessa "label specie" 
def scatter_plot_result(df):
    # Aggiungiamo un rumore ai valori di "type" e "predict_label" così da non avere sovrapposizione dei punti nel grafico
    df['type'] = df['type'].apply(lambda x: x + np.random.randint(-40,40)/100)
    df['predict_label'] = df['predict_label'].apply(lambda x: x + np.random.randint(-40,40)/100)
    # eseguiamo il raggruppamento su base della predizione
    groups = df.groupby('predict')
    fig = plt.figure()
    ax = fig.gca()
    # Grafico della diagonale di "corretta" assegnazione tramite MV, ovvero gli elementi che sono
    # dentro alla circonferenza tramite metodo del MV sono stati assegnati alla stessa specie originale
    circle0 = plt.Circle((0,0), 0.5,  fill=False)
    circle1 = plt.Circle((1,1), 0.5,  fill=False)
    circle2 = plt.Circle((2,2), 0.5,  fill=False)
    circle3 = plt.Circle((3,3), 0.5,  fill=False)
    circle4 = plt.Circle((4,4), 0.5,  fill=False)
    circle5 = plt.Circle((5,5), 0.5,  fill=False)
    circle6 = plt.Circle((6,6), 0.5,  fill=False)

    ax.set_xticks(np.arange(-1, 7, 1))
    ax.set_yticks(np.arange(-1, 7, 1))
    # Scatterplot dei punti type vs predict_label con etichetta basata sul cluster assegnato dall'algoritmo di clusterizzazione
    for cluster, group in groups:
        plt.plot(group.type, group.predict_label, marker='o', linestyle='', markersize=8, label=cluster)
    
    plt.xlabel('type')
    plt.ylabel('majority voting label')
    # Plotto la "diagonale corretta"
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

# Come scatter_plot_result per ogni modello presente nella lista df_all esegue lo scatter plot della specie originale (asse x) vs specie assegnata per MV (asse y)
# assegnando un colore diverso per ogni cluster di appartenenza
# Se raggiugne il limite di sei scatterplot per la stessa figura (2x3) genera una nuova figura
def scatter_plot_all(df_all,colum_all):
    cnt=1;
    fig = plt.figure()
        
            
    for df_n in df_all:
        df=df_n[colum_all].copy()
        # Aggiungiamo un rumore ai valori di "type" e "predict_label" così da non avere sovrapposizione dei punti nel grafico
        df['type'] = df['type'].apply(lambda x: x + np.random.randint(-40,40)/100)
        df['predict_label'] = df['predict_label'].apply(lambda x: x + np.random.randint(-40,40)/100)
        groups = df.groupby('predict')
        if cnt>6:
            plt.show()
            cnt=1;
            fig = plt.figure()
    
        plt.subplot(2,3,cnt)
        ax = fig.gca()
        # Grafico della diagonale di "corretta" assegnazione tramite MV, ovvero gli elementi che sono
        # dentro alla circonferenza tramite metodo del MV sono stati assegnati alla stessa specie originale
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
            # Scatterplot dei punti type vs predict_label con etichetta basata sul cluster assegnato dall'algoritmo di clusterizzazione
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
        cnt=cnt+1
    plt.show()
    return(df)
# Computo dell RAND index tramiter la funzione di sklearn
def rand_index(df):
    ri = rand_score(df['type'], df['predict'])
    return(ri)

# Computo dell Adjust RAND index tramiter la funzione di sklearn
def adjusted_rand_index(df):
    ari = adjusted_rand_score(df['type'], df['predict'])
    return(ari)

# Computo del mutual information e la sua versione normalizzata tramiter la funzione di sklearn
def mutual_information_index(df):
    MI = mutual_info_score(df['type'], df['predict'])
    NMI = normalized_mutual_info_score(df['type'], df['predict'])
    return(MI,NMI)
# Computo dell'omogeneità, completezza e v_index tramiter la funzione di sklearn
def v_measure_index(df):
    HS = homogeneity_score(df['type'], df['predict'])
    CS = completeness_score(df['type'], df['predict'])
    V = v_measure_score(df['type'], df['predict'], beta=1.0)
    return(HS,CS,V)

# # Computo della siluet tramiter la funzione di sklearn
def silhouette_score_index(df):
    ss = silhouette_score(df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']], df['predict'])
    ss=(ss+1)/2
    return(ss)

# Funzione che riceve in ingresso il dataframe a valle di una clusterizzazione e il dataframe "model_comparison"
# in cui saranno memorizzate le performance di tale modello tramite i vari indici
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
    # ari_score=adjusted_rand_index(df)
    
    # Mutual Information (MI, NMI)
    # Le informazioni reciproche (MI, NMI, AMI) misurano l'accordo tra le assegnazioni del cluster.
    # Un punteggio più alto indica una somiglianza maggiore.
    MI_score,NMI_score=mutual_information_index(df)
    
    # V-measure
    # V-Measure misura la correttezza delle assegnazioni del cluster usando l'analisi dell'entropia condizionale.
    # Un punteggio più alto indica una somiglianza maggiore
    # Omogeneità: ogni cluster contiene solo membri di una singola classe (un po 'come "precisione")
    # Completezza: tutti i membri di una determinata classe sono assegnati allo stesso cluster (un po 'come "richiamo")
    HS_score,CS_score,V_score=v_measure_index(df)
    
    # Silhouette Score aka Silhouette Coefficient
    # Silhouette score aka Silhouette Coefficient is an evaluation metric that results in the range of -1 to 1. A score near 1 signifies the best importance that the data point is very compact within the cluster to which it belongs and far away from the other clusters. The score near -1 signifies the least or worst importance of the data point. A score near 0 signifies overlapping clusters. 
    ss_score=silhouette_score_index(df_full)
    
    # Salva tutti i valori dei singoli indici in una nuova riga che andrà aggiunta al dataframe "model_comparison"
    new_row=[[df['model_type'][0],df['model_name'][0],ri_score,MI_score,NMI_score,HS_score,CS_score,V_score,ss_score]]
    
    model_comparison= model_comparison.append(pd.DataFrame(new_row, columns=model_comparison.columns))
    return(model_comparison)

# Algoritmo di Clustering Affinity Propagation (AP)
def AffinityM_prediction(df, k):
    # Costruzione del "modello" tramite la libreria sklearn con i valori degli iperparametri ricevuti in ingresso alla funzione
    affinityM = AffinityPropagation(damping=k)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Addestramento modello sul dataset di training
    affinityM.fit(features)
    # Salvataggio delle previsioni
    df['predict'] = affinityM.predict(features)
    df['model_type'] = 'AffinityPropagation'
    df['model_name'] = 'AffinityPropagation_'+str(k)
    majority_voting_label(df)
    return(df)

# Algoritmo di Agglomerative Clustering
def agglomerativeClustering_prediction(df, k):
    # Costruzione del "modello" tramite la libreria sklearn con i valori degli iperparametri ricevuti in ingresso alla funzione
    agglomerativeC = AgglomerativeClustering(n_clusters=k)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Addestramento modello sul dataset di training con assegnazione delle labels
    df['predict'] = agglomerativeC.fit_predict(features)
    df['model_type'] = 'AgglomerativeClustering'
    df['model_name'] = 'AgglomerativeClustering_'+str(k)
    majority_voting_label(df)
    return(df)
   
# Algoritmo di BIRCH
def birch_prediction(df, k,th):
    # Costruzione del "modello" tramite la libreria sklearn con i valori degli iperparametri ricevuti in ingresso alla funzione
    birchM = Birch(threshold=th, n_clusters=k)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Addestramento modello sul dataset di training
    birchM.fit(features)
    # Salvataggio delle previsioni
    df['predict'] = birchM.predict(features)
    df['model_type'] = 'birch'
    df['model_name'] = 'birchMlustering_k:_'+str(k)+'_th_'+str(th)
    majority_voting_label(df)
    return(df)


# Algoritmo di DBSCAN
def dbscan_prediction(df,eps_v,min_sam):
    # Costruzione del "modello" tramite la libreria sklearn con i valori degli iperparametri ricevuti in ingresso alla funzione
    dbscan_model = DBSCAN( eps = eps_v, min_samples = min_sam)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Addestramento modello sul dataset di training
    dbscan_model.fit(features)    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = dbscan_model.labels_
    df['model_type'] = 'dbscan'
    df['model_name'] = 'dbscan_eps_'+str(eps_v)+'_min_samples_'+str(min_sam)
    majority_voting_label(df)
    return(df)

# Algoritmo OPTICS
def optics_m_prediction(df,eps_v,min_sam):
    # Costruzione del "modello" tramite la libreria sklearn con i valori degli iperparametri ricevuti in ingresso alla funzione
    optics_m_model = OPTICS( eps = eps_v, min_samples = min_sam)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    # Addestramento modello sul dataset di training con assegnazione delle labels
    df['predict'] = optics_m_model.fit_predict(features)    
    df['model_type'] = 'optics'
    df['model_name'] = 'optics_m_eps_'+str(eps_v)+'_min_samples_'+str(min_sam)
    majority_voting_label(df)
    return(df)

# Algoritmo di Clustering K-means
def kmeans_prediction(df, k):
    # Costruzione del "modello" tramite la libreria sklearn con i valori degli iperparametri ricevuti in ingresso alla funzione
    kmeans = KMeans(n_clusters=k)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Addestramento modello sul dataset di training
    kmeans.fit(features)
    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = kmeans.predict(features)
    df['model_type'] = 'kmeans'
    df['model_name'] = 'kmeans_'+str(k)
    majority_voting_label(df)
    return(df)

# Algoritmo di Clustering Mini-Batch K-Means
def mb_kmeans_prediction(df, k):
    # Costruzione del "modello" tramite la libreria sklearn con i valori degli iperparametri ricevuti in ingresso alla funzione
    mb_kmeans = MiniBatchKMeans(n_clusters=k)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Addestramento modello sul dataset di training
    mb_kmeans.fit(features)
    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = mb_kmeans.predict(features)
    df['model_type'] = 'mb_kmeans'
    df['model_name'] = 'mb_kmeans_'+str(k)
    majority_voting_label(df)
    return(df)


# Algoritmo di Clustering Mean shift
def mean_shift_prediction(df):
    # Costruzione del "modello" tramite la libreria sklearn coni valori degli iperparametri ricevuti in ingresso alla funzione
    mean_shift = MeanShift()
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()
    # Addestramento modello sul dataset di training
    mean_shift.fit(features)
    
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = mean_shift.predict(features)
    df['model_type'] = 'mean_shift'
    df['model_name'] = 'mean_shift'
    majority_voting_label(df)
    return(df)

# Algoritmo di Spectral Clustering
def spectral_clustering_prediction(df,k):
    # Costruzione del "modello" tramite la libreria sklearn coni valori degli iperparametri ricevuti in ingresso alla funzione
    spectral_clustering = SpectralClustering(n_clusters=k)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()    
    
    # Addestramento modello sul dataset di training ed assegnazione delle etichette
    df['predict'] = spectral_clustering.fit_predict(features)
    df['model_type'] = 'spectral_clustering'
    df['model_name'] = 'spectral_clustering'+str(k)
    majority_voting_label(df)
    return(df)

# Algoritmo di Clustering Gaussian Mixture
def gaussian_mixture_prediction(df,k):
    # Costruzione del "modello" tramite la libreria sklearn coni valori degli iperparametri ricevuti in ingresso alla funzione
    gaussian_mixture =GaussianMixture(n_components=k)
    # Creazione dataframe di training
    features=df[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']].copy()    
    # Addestramento modello sul dataset di training
    gaussian_mixture.fit(features)
    # Assegna le etichette dei cluster a ogni oggetto del dataset
    df['predict'] = gaussian_mixture.predict(features)
    df['model_type'] = 'gaussian_mixture'
    df['model_name'] = 'gaussian_mixture'+str(k)
    majority_voting_label(df)
    return(df)

# Riceve in ingresso il dataframe "model_comperison" con i risultati in termini di indici delle singole varianti (combinazione iperparametri)
# per ogni algoritmo di clustering ed estrai i migliori (per algoritmo) sulla mase del "resume_index"
def get_best_models(model_comparison):
    df=model_comparison.copy()
    # Computa il resume index come media degli indici di rand index, normalized mutual info, v_measure e siluetteN
    df['resume_index']=df[['ri','NMI','V','siluetteN']].mean(axis=1)
    # raggruppa il dataframe per tipologia di algoritmo di clustering e ne estrae gli indici (puntatori di riga  del dataframe)
    # dei migliori rappresentanti con cui successivamente estrae un sottodataframe con i best_models
    df.groupby('model_type')['resume_index'].max()
    idx_best = df.groupby('model_type')['resume_index'].transform(max) == df['resume_index']
    best_models=df[idx_best]
    return(best_models)




# def barplot_class_feature_distribution(df,species_dict):
#    feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
#    cnt=0
#    sns.set() 
#    fig, axes = plt.subplots(2, 4, sharey=True)
#    fig.suptitle('Class features composition')
#    for feature in feature_name:
#        sub_df=df.groupby(["type", feature])[feature].count().reset_index(name="count").copy()
#        sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
#        sub_df['type']=sub_df['type'].map(species_dict) 
#        sns.barplot(x ="type", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
#        axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)
#        # axes[cnt//4,cnt%4].set_title(feature)       
#        cnt=cnt+1
    
#    feature_name=['backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
#    cnt=0
#    fig, axes = plt.subplots(2, 4, sharey=True)
#    fig.suptitle('Class features composition')
#    for feature in feature_name:
#         sub_df=df.groupby(["type", feature])[feature].count().reset_index(name="count").copy()
#         sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
#         sub_df['type']=sub_df['type'].map(species_dict) 
#         sns.barplot(x ="type", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
#         axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)        
#         # axes[cnt//4,cnt%4].set_title(feature)        
#         cnt=cnt+1


# Barplot della distribuzione percentuale all'interno di ogni cluster dei valori delle singole features
# lo scopo era fare un raffronto percentuale tra il cluster originale e quello otte nuto dagli algoritmi (NON usato alla fine)
# N.B.: NOT USED
# def barplot_class_feature_percentage(df,species_dict):
#     feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','type']
#     for result_predict in (df['predict'].unique()):
        
#         sub_df=df[df["predict"]==result_predict].copy() 
#         cnt=1
#         plt.figure()
#         for feature in feature_name:
#             # get colum value distribution
#             a = (sub_df[feature].value_counts(normalize=True) 
#                         .mul(100)
#                         .rename_axis(feature)
#                         .reset_index(name='percentage'))
#             # creating the bar plot
            
#             plt.subplot(3, 6,cnt)
#             #plt.title(feature)
#             plt.bar(a[feature].values.tolist(), a['percentage'], align='center', color ='b')
         
#             plt.xlabel(feature)
#             plt.ylabel("Percentage [%]")    
#             #ax[cnt].plt.show()
#             cnt=cnt+1
#         # Show the plots
#         sub_df['predict']=sub_df['predict'].map(species_dict) 
#         plt.suptitle("predict:"+sub_df['predict'].unique())
#         plt.show()

# def barplot_class_feature_comparison(df,species_dict, major_label_voting):
#    if major_label_voting==1:
#         predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
#         predict_major=df.set_index('predict').to_dict()['type']
#         for k in predict_major:
#             predict_major[k]=str(k)+'_'+species_dict[predict_major[k]]
        
#    feature_name=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed']
#    cnt=0
#    sns.set() 
#    fig, axes = plt.subplots(2, 4, sharey=True)
#    fig.suptitle('Class features composition')
#    for feature in feature_name:
#        sub_df=df.groupby(["predict", feature])[feature].count().reset_index(name="count")
#        sub_df['perc'] = sub_df['count'].groupby(sub_df['predict']).transform(lambda x: x/x.sum())
#        if major_label_voting==1:           
#            sub_df['predict']=sub_df['predict'].map(predict_major)
#        sns.barplot(x ="predict", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
#        axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)
#        # axes[cnt//4,cnt%4].set_title(feature)       
#        cnt=cnt+1
    
#    feature_name=['backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
#    cnt=0
#    fig, axes = plt.subplots(2, 4, sharey=True)
#    fig.suptitle('Class features composition')
#    for feature in feature_name:
#         sub_df=df.groupby(["predict", feature])[feature].count().reset_index(name="count")
#         sub_df['perc'] = sub_df['count'].groupby(sub_df['predict']).transform(lambda x: x/x.sum())
#         if major_label_voting==1:           
#             sub_df['predict']=sub_df['predict'].map(predict_major)
#         sns.barplot(x ="predict", y = 'perc', data = sub_df, hue = feature,ax=axes[cnt//4,cnt%4])
#         axes[cnt//4,cnt%4].set_xticklabels(axes[cnt//4,cnt%4].get_xticklabels(), rotation=45)        
#         # axes[cnt//4,cnt%4].set_title(feature)        
#         cnt=cnt+1

# funzione per avere una visone tramite barplot delle performance in percentuale del singolo algoritmo nella clusterizzazione
# N.B. NOT USED
# def classification_percentage_performance(df,species_dict):
#     df=replace_predict_with_major(df).copy()    
#     sub_df=df.groupby(["type", "predict_label"])["predict"].count().reset_index(name="count")
#     sub_df['perc'] = sub_df['count'].groupby(sub_df['type']).transform(lambda x: x/x.sum())
#     if   type(sub_df['predict'][0]) == int or type(sub_df['predict'][0]) == float:
#         sub_df['predict']=sub_df['predict'].map(species_dict)
#     if   type(sub_df['type'][0]) == int or type(sub_df['type'][0]) == float:
#         sub_df['type']=sub_df['type'].map(species_dict)
#     sns.set() 
#     fig, axes = plt.subplots(2, 1)
#     fig.suptitle('Distribution of classified classes')
#     sns.barplot(x ="type", y = 'perc', data = sub_df, hue = "predict",ax=axes[0])    
#     sns.barplot(x ="type", y = 'count', data = sub_df, hue = "predict",ax=axes[1])
    

# Indice di Similarità all'interno dei singoli cluster
# N.B.: NOT USED
# def similarity_index(df):
#     # Assegnazione della label di maggioranza al cluster
#     predict_major=df['type'].groupby(df['predict']).value_counts().groupby(level=[0], group_keys=False).head(1).to_frame('counts').reset_index()
#     predict_major=df.set_index('predict').to_dict()['type']
#     if   type(df['predict'][0]) == int or type(df['predict'][0]) == float:
#         df['predict']=df['predict'].map(predict_major)
    

#     df=replace_predict_with_major(df)
#     similarity_dict={}
#     for real_type in (df['type'].unique()):
#         sub_df=df[df["type"]==real_type]
#         match=sub_df[sub_df["predict"]==real_type]
#         if match.shape[0]>0:
#             similarity_dict[real_type]=match.shape[0]/sub_df.shape[0]*100
#         else:
#             similarity_dict[real_type]=0
#     myKeys = list(similarity_dict.keys())
#     myKeys.sort()
#     sorted_dict = {i: similarity_dict[i] for i in myKeys}
#     similarity_dict=sorted_dict
    
#     fig = plt.figure()
#     ax = fig.gca()
#     plt.bar(range(len(similarity_dict)), list(similarity_dict.values()), tick_label=list(similarity_dict.keys()))
#     plt.xlabel('type')
#     plt.ylabel('Similarity inside the group')
#     plt.show()
#     return(similarity_dict)

# Indice di Dissimilarità all'interno dei singoli cluster
# N.B.: NOT USED
# def dissimilarity_index(df):
#     df_tmp = pd.DataFrame(columns = list(df.columns.values)).copy()
#     # Valutazione predict
#     for result_predict in (df['predict'].unique()):
#         sub_df=df[df["predict"]==result_predict]
#         major_species=sub_df['type'].value_counts().idxmax()
#         sub_df['predict label'] =major_species
#         df_tmp=pd.concat([df_tmp, sub_df], axis=0)

#     df=df_tmp
#     dissimilarity_dict={}
#     for real_type in (df['type'].unique()):
#         sub_df=df[df["type"]==real_type]
#         match=sub_df[sub_df["predict label"]==real_type]
#         dissimilarity_dict[real_type]=((sub_df.shape[0]-match.shape[0])/sub_df.shape[0])*100
#     myKeys = list(dissimilarity_dict.keys())
#     myKeys.sort()
#     sorted_dict = {i: dissimilarity_dict[i] for i in myKeys}
#     dissimilarity_dict=sorted_dict
    
#     fig = plt.figure()
#     ax = fig.gca()
#     plt.bar(range(len(dissimilarity_dict)), list(dissimilarity_dict.values()), tick_label=list(dissimilarity_dict.keys()))
#     plt.xlabel('type')
#     plt.ylabel('Dissimilarity inside the group')
#     plt.show()
#     return(dissimilarity_dict)
