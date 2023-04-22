import support_function as sf
import pandas as pd
import os
# colnames = ['nome animale', 'pelo','piume','uova','latte','aereo','acquatico','predatore','dentato','spina dorsale','respira','velenoso ','pinne','zampe','coda','domestico','tagliagatto','tipo']
colnames = ['animal name', 'hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','type']
zoo_data = pd.read_csv(os.getcwd()+'/Dataset/zoo.data', delimiter=',', names=colnames, header=None)
zoo_data['type']=zoo_data['type']-1
print(zoo_data)

# Costruiamo il dizionario delle Specie
species_dict =dict(list(enumerate(['mammiferi', 'uccelli', 'rettili', 'pesci', 'anfibi', 'insetti','invertebrati'])))
print(species_dict)


# Plotting the heatmap for correlation between features and class
sf.feature_vs_class_heatmap(zoo_data,species_dict)

# Dataset composition analysis
sf.class_distributuion_barplot(zoo_data,species_dict)
sf.barplot_class_feature_distribution(zoo_data,species_dict)

# create models index dataframe
model_comparison = pd.DataFrame(columns=['model_type','model', 'ri','MI','NMI','HS','CS','V','siluetteN'])

# AffinityPropagation
for k in range(5,10,1):
    k=k/10
    affinityM=sf.AffinityM_prediction(zoo_data.copy(),k)
    result_affinityM=affinityM[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_affinityM=sf.replace_species(result_affinityM,species_dict)
    #sf.confusion_matrix_plot(result_affinityM)
    #scatter_df=sf.scatter_plot_result(affinityM[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(affinityM,result_affinityM,model_comparison)

# Agglomerative Clustering
for k in range(5,10,1):
    agglomerativeC=sf.agglomerativeClustering_prediction(zoo_data.copy(),k)
    result_agglomerativeC=agglomerativeC[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_agglomerativeC=sf.replace_species(result_agglomerativeC,species_dict)
    #sf.confusion_matrix_plot(result_agglomerativeC)
    #scatter_df=sf.scatter_plot_result(agglomerativeC[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(agglomerativeC,result_agglomerativeC,model_comparison)
    
# BIRCH
for k in range(5,10,1):
    for th in range(1,5,2):
        th=th/100
        birchM=sf.birch_prediction(zoo_data.copy(),k,th)
        result_birchM=birchM[['animal name','model_type','model_name','type','predict_label','predict']].copy()
        result_birchM=sf.replace_species(result_birchM,species_dict)
        #sf.confusion_matrix_plot(result_birchM)
        #scatter_df=sf.scatter_plot_result(birchM[['animal name','model_type','model_name','type','predict_label','predict']].copy())
        model_comparison=sf.model_performance_evaluation(birchM,result_birchM,model_comparison)


## DBSCAN
for n in range(1,10,2):
    dbscan=sf.dbscan_prediction(zoo_data.copy(),0.75,n)
    result_dbscan=dbscan[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_dbscan=sf.replace_species(result_dbscan,species_dict)
    # Scatter plot
    #scatter_df=sf.scatter_plot_result(dbscan[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(dbscan,result_dbscan,model_comparison)

# OPTICS
for n in range(2,10,2):
    optics_m=sf.optics_m_prediction(zoo_data.copy(),0.75,n)
    result_optics_m=optics_m[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_optics_m=sf.replace_species(result_optics_m,species_dict)
    #sf.confusion_matrix_plot(result_optics_m)
    #scatter_df=sf.scatter_plot_result(optics_m[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(optics_m,result_optics_m,model_comparison)
    
# k-means algorithm
for k in range(4,10,1):
    kmeans=sf.kmeans_prediction(zoo_data.copy(),k)
    result_kmeans=kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_kmeans=sf.replace_species(result_kmeans,species_dict)
    #sf.confusion_matrix_plot(result_kmeans)
    #scatter_df=sf.scatter_plot_result(kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(kmeans,result_kmeans,model_comparison)

# Mini-Batch k-means algorithm
for k in range(4,10,1):
    mb_kmeans=sf.mb_kmeans_prediction(zoo_data.copy(),k)
    result_mb_kmeans=mb_kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_mb_kmeans=sf.replace_species(result_mb_kmeans,species_dict)
    #sf.confusion_matrix_plot(result_mb_kmeans)
    #scatter_df=sf.scatter_plot_result(mb_kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(mb_kmeans,result_mb_kmeans,model_comparison)

# Mean shift algorithm
mean_shift=sf.mean_shift_prediction(zoo_data.copy())
result_mean_shift=mean_shift[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_mean_shift=sf.replace_species(result_mean_shift,species_dict)
#sf.confusion_matrix_plot(result_mean_shift)
#scatter_df=sf.scatter_plot_result(mean_shift[['animal name','model_type','model_name','type','predict_label','predict']].copy())
model_comparison=sf.model_performance_evaluation(mean_shift,result_mean_shift,model_comparison)

# Spectral Clustering
for k in range(4,10,1):
    spectral_clustering=sf.spectral_clustering_prediction(zoo_data.copy(),k)
    result_spectral_clustering=spectral_clustering[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_spectral_clustering=sf.replace_species(result_spectral_clustering,species_dict)
    #sf.confusion_matrix_plot(result_spectral_clustering)
    #scatter_df=sf.scatter_plot_result(spectral_clustering[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(spectral_clustering,result_spectral_clustering,model_comparison)
    

# Gaussian Mixture 
for k in range(4,10,1):
    gaussian_mixture=sf.gaussian_mixture_prediction(zoo_data.copy(),k)
    result_gaussian_mixture=gaussian_mixture[['animal name','model_type','model_name','type','predict_label','predict']].copy()
    result_gaussian_mixture=sf.replace_species(result_gaussian_mixture,species_dict)
    #sf.confusion_matrix_plot(result_gaussian_mixture)
    #scatter_df=sf.scatter_plot_result(gaussian_mixture[['animal name','model_type','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(gaussian_mixture,result_gaussian_mixture,model_comparison)

# Best model selection
best_models=sf.get_best_models(model_comparison)
#sf.feature_vsclustered_class(zoo_data,agglomerativeC,species_dict)


##
# AffinityPropagation
k=0.5
affinityM=sf.AffinityM_prediction(zoo_data.copy(),k)
result_affinityM=affinityM[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_affinityM=sf.replace_species(result_affinityM,species_dict)
sf.confusion_matrix_plot(result_affinityM)
scatter_df=sf.scatter_plot_result(affinityM[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,affinityM,species_dict)

# Agglomerative Clustering
k=9
agglomerativeC=sf.agglomerativeClustering_prediction(zoo_data.copy(),k)
result_agglomerativeC=agglomerativeC[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_agglomerativeC=sf.replace_species(result_agglomerativeC,species_dict)
sf.confusion_matrix_plot(result_agglomerativeC)
scatter_df=sf.scatter_plot_result(agglomerativeC[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,agglomerativeC,species_dict)

# BIRCH
k=9
th=0.01
birchM=sf.birch_prediction(zoo_data.copy(),k,th)
birchM9=birchM.copy()

result_birchM=birchM[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_birchM=sf.replace_species(result_birchM,species_dict)
result_birchM9=result_birchM.copy()
sf.confusion_matrix_plot(result_birchM)
scatter_df=sf.scatter_plot_result(birchM[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,birchM,species_dict)

# BIRCH
th=0.03
birchM=sf.birch_prediction(zoo_data.copy(),k,th)
result_birchM=birchM[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_birchM=sf.replace_species(result_birchM,species_dict)
sf.confusion_matrix_plot(result_birchM)
scatter_df=sf.scatter_plot_result(birchM[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,birchM,species_dict)

## DBSCAN
n=1
dbscan=sf.dbscan_prediction(zoo_data.copy(),0.75,n)
result_dbscan=dbscan[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_dbscan=sf.replace_species(result_dbscan,species_dict)
sf.confusion_matrix_plot(result_dbscan)
scatter_df=sf.scatter_plot_result(dbscan[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,dbscan,species_dict)

##
dbscan_partial_fish=dbscan[dbscan["predict_label"]==3].copy()
dbscan_partial_fish = dbscan_partial_fish.drop(['model_type','model_name','type','predict_label'], axis=1)
print(dbscan_partial_fish)

# OPTICS
n=8
optics_m=sf.optics_m_prediction(zoo_data.copy(),0.75,n)
result_optics_m=optics_m[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_optics_m=sf.replace_species(result_optics_m,species_dict)
sf.confusion_matrix_plot(result_optics_m)
scatter_df=sf.scatter_plot_result(optics_m[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,optics_m,species_dict)

# k-means algorithm
k=8
kmeans=sf.kmeans_prediction(zoo_data.copy(),k)
result_kmeans=kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_kmeans=sf.replace_species(result_kmeans,species_dict)
sf.confusion_matrix_plot(result_kmeans)
scatter_df=sf.scatter_plot_result(kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,kmeans,species_dict)

# Mini-Batch k-means algorithm
k=8
mb_kmeans=sf.mb_kmeans_prediction(zoo_data.copy(),k)
result_mb_kmeans=mb_kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_mb_kmeans=sf.replace_species(result_mb_kmeans,species_dict)
sf.confusion_matrix_plot(result_mb_kmeans)
scatter_df=sf.scatter_plot_result(mb_kmeans[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,mb_kmeans,species_dict)

# Mean shift algorithm
mean_shift=sf.mean_shift_prediction(zoo_data.copy())
result_mean_shift=mean_shift[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_mean_shift=sf.replace_species(result_mean_shift,species_dict)
sf.confusion_matrix_plot(result_mean_shift)
scatter_df=sf.scatter_plot_result(mean_shift[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,mean_shift,species_dict)

# Spectral Clustering
k=9
spectral_clustering=sf.spectral_clustering_prediction(zoo_data.copy(),k)
result_spectral_clustering=spectral_clustering[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_spectral_clustering=sf.replace_species(result_spectral_clustering,species_dict)
sf.confusion_matrix_plot(result_spectral_clustering)
scatter_df=sf.scatter_plot_result(spectral_clustering[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,spectral_clustering,species_dict)

# Gaussian Mixture 
k=7
gaussian_mixture=sf.gaussian_mixture_prediction(zoo_data.copy(),k)
result_gaussian_mixture=gaussian_mixture[['animal name','model_type','model_name','type','predict_label','predict']].copy()
result_gaussian_mixture=sf.replace_species(result_gaussian_mixture,species_dict)
sf.confusion_matrix_plot(result_gaussian_mixture)
scatter_df=sf.scatter_plot_result(gaussian_mixture[['animal name','model_type','model_name','type','predict_label','predict']].copy())
sf.feature_vsclustered_class(zoo_data,gaussian_mixture,species_dict)

# Scatter plot all
sf.scatter_plot_all([affinityM,agglomerativeC,birchM9, birchM,dbscan,optics_m,kmeans,mb_kmeans,mean_shift,spectral_clustering,gaussian_mixture],['animal name','model_type','model_name','type','predict_label','predict'])

# # Where is platypus?
# platypus_result=pd.concat([result_affinityM, result_agglomerativeC,result_birchM9, result_birchM,result_dbscan,result_optics_m,result_kmeans,result_mb_kmeans,result_mean_shift,result_spectral_clustering,result_gaussian_mixture], axis=0)
# platypus_result=platypus_result[platypus_result["animal name"]=='platypus']
# predict_platypus=pd.DataFrame(platypus_result['predict_label'].value_counts())

# plt.figure()
# plt.title('Where is platypus?')
# plt.bar(predict_platypus.index, predict_platypus['predict_label'], align='center')     
# plt.ylabel("Count")   
# plt.show()
 

