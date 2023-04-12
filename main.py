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


# k-means algorithm
kmeans=sf.kmeans_prediction(zoo_data,7)
# Mostra le prime 20 righe del dataframe con la colonna Cluster aggiunta
kmeans.head(20)

# DBSCAN
dbscan=sf.dbscan_prediction(zoo_data)


## Valutazione cluster
# Labelize result
result_kmeans=zoo_data[['animal name','type','predict']].copy()
result_kmeans=sf.replace_species(result_kmeans,species_dict)

# Classification performance
sf.classification_percentage_performance(result_kmeans,species_dict)
 # Similarity index
similarity_kmeans=sf.similarity_index(result_kmeans)
# Dissimilarità index
dissimilarity_kmeans=sf.dissimilarity_index(result_kmeans)

# Adjusted Rand Index
#The adjusted rand index is an evaluation metric that is used to measure the similarity between two clustering by considering all the pairs of the n_samples and calculating the counting pairs of the assigned in the same or different clusters in the actual and predicted clustering.  
#The adjusted rand index score is defined as:
#ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
# A score above 0.7 is considered to be a good match. 
ari_kmeans=sf.adjusted_rand_index(result_kmeans)

#Rand Index
#The Rand index is different from the adjusted rand index. Rand index does find the similarity between two clustering by considering all the pairs of the n_sample but it ranges from 0 to 1. whereas ARI ranges from -1 to 1. 
#The rand index is defined as:
# RI = (number of agreeing pairs) / (number of pairs)
ri_kmeans=sf.rand_index(result_kmeans)

# Silhouette Score aka Silhouette Coefficient
# Silhouette score aka Silhouette Coefficient is an evaluation metric that results in the range of -1 to 1. A score near 1 signifies the best importance that the data point is very compact within the cluster to which it belongs and far away from the other clusters. The score near -1 signifies the least or worst importance of the data point. A score near 0 signifies overlapping clusters. 
ss_kmeans=sf.silhouette_score_index(kmeans)

# Scatter plot
scatter_df=sf.scatter_plot_result(kmeans[['animal name','type','predict']])     
      
# bar plot of predicted feature class composition
# sf.barplot_class_feature_percentage(zoo_data,species_dict)
sf.barplot_class_feature_comaprison(kmeans,species_dict)