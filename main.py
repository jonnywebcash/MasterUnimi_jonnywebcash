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
model_comparison = pd.DataFrame(columns=['model', 'ri','ari','MI','NMI','AMI','HS','CS','V','siluette'])

# k-means algorithm
for k in range(4,10,1):
    kmeans=sf.kmeans_prediction(zoo_data.copy(),k)
    result_kmeans=kmeans[['animal name','model_name','type','predict_label','predict']].copy()
    result_kmeans=sf.replace_species(result_kmeans,species_dict)
    scatter_df=sf.scatter_plot_result(kmeans[['animal name','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(kmeans,result_kmeans,model_comparison)


## DBSCAN
for n in range(1,10,2):
    dbscan=sf.dbscan_prediction(zoo_data.copy(),0.75,n)
    result_dbscan=dbscan[['animal name','model_name','type','predict_label','predict']].copy()
    result_dbscan=sf.replace_species(result_dbscan,species_dict)
    # Scatter plot
    scatter_df=sf.scatter_plot_result(dbscan[['animal name','model_name','type','predict_label','predict']].copy())
    model_comparison=sf.model_performance_evaluation(dbscan,result_dbscan,model_comparison)



# Classification performance
# Percentage_true_cluster: indice che valuta per m
sf.classification_percentage_performance(result_kmeans,species_dict)
sf.barplot_class_feature_comparison(dbscan,species_dict,1)


##
# https://towardsdatascience.com/7-evaluation-metrics-for-clustering-algorithms-bdc537ff54d2
##

