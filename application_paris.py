import shapefile
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import StratifiedShuffleSplit
import requests
from joblib import dump, load
import pickle 
import sys

# -------------------------------------------------------------------------------
def read_shapefile(sf):
    """
    Shapefile to Pandas dataframe with a 'coords' column
    Source : https://gist.github.com/aerispaha/f098916ac041c286ae92d037ba5c37ba
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

def centroid_x(liste) :
    """
    Transform Polynom coordinates into a single point : Centroid
    Only the x coordinate is returned
    """
    x_lon = np.zeros((len(liste),1))
    for ip in range(len(liste)):
        x_lon[ip] = liste[ip][0]
    x_centroid = x_lon.mean()
    return x_centroid

def centroid_y(liste) :
    """
    Transform Polynom coordinates into a single point : Centroid
    Only the y coordinate is returned
    """
    y_lat = np.zeros((len(liste),1))
    for ip in range(len(liste)):
        y_lat[ip] = liste[ip][1]
    y_centroid = y_lat.mean()
    return y_centroid

def add(a, b):
    """
    Function to obtain the concatenation of a and b, with b having a size of 4.
    Transformation required for the creation of the id (parcelles.shp dataframe)
    """
    b = b.zfill(4)
    return a + b

# -------------------------------------------------------------------------------
def main():
    option = sys.argv[1]
    if option == 'train':
        """
        Function to realize the data preparation and the training of the KNeighborsRegressor model
        Source : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        Source : https://hureauxarnaud.medium.com/projet-estimateur-de-prix-dun-bien-immobilier-bas%C3%A9-sur-du-machine-learning-ae578fdacaca
        """
        print("Reading file parcelles.shp")
        sf = shapefile.Reader('parcelles.shp')
        shapefile_df = read_shapefile(sf)
        shapefile_df["x"] = shapefile_df["coords"].apply(centroid_x)
        shapefile_df["y"] = shapefile_df["coords"].apply(centroid_y)
        print("Backup of the dataframe shapefile_df in pickle format")
        shapefile_df.to_pickle("./shapefile_df.pkl")
        
        print("Reading file valeursfoncieres-2020.txt")
        data = pd.read_csv('../valeursfoncieres-2020.txt', sep="|", low_memory=False)
        
        print("Pretreatment file valeursfoncieres-2020.txt")
        data_paris = data[data["Code departement"]=='75'].copy()
        data_paris = data_paris[data_paris["Type local"] == "Appartement"]
        
        liste_drop = ['Code service CH','Reference document','1 Articles CGI','2 Articles CGI', '3 Articles CGI', '4 Articles CGI', '5 Articles CGI', \
                    'Date mutation','B/T/Q', '1er lot', 'Surface Carrez du 1er lot', '2eme lot', 'Surface Carrez du 2eme lot', '3eme lot', 'Surface Carrez du 3eme lot', \
                    '4eme lot','Surface Carrez du 4eme lot', '5eme lot', 'Surface Carrez du 5eme lot', 'Code type local','Identifiant local','Nature culture', \
                    'Nature culture speciale', 'Surface terrain', 'No Volume', 'Prefixe de section']
        data_paris = data_paris.drop(columns=liste_drop)
        
        data_paris = data_paris[data_paris['Nombre de lots']<2]
        data_paris["Code postal"] = data_paris["Code postal"].astype(int)
        data_paris['Valeur fonciere'] = data_paris['Valeur fonciere'].str.replace(",", ".").astype(float)
        data_paris = data_paris[data_paris['Nature mutation']=='Vente'].copy()
        data_paris = data_paris.drop_duplicates(keep='first')
        
        liste_group = list(data_paris.columns)
        liste_group.remove("Surface reelle bati")
        liste_group.remove('Valeur fonciere')

        # Allow to remove duplicates with a "Surface reelle bati" and a 'Valeur fonciere' differents for a same adress, code plan ...
        data_paris = data_paris.groupby(liste_group).mean().reset_index() 
        data_paris['prix_m²'] = data_paris['Valeur fonciere']/data_paris['Surface reelle bati'].replace(",", ".").astype(float)
        
        # Firsts Outliers removed based on prices found on the Internet
        # Source : https://immobilier.lefigaro.fr/article/a-paris-le-metre-carre-peut-valoir-plus-de-50-000-euros_cf0a892c-1f1b-11e9-b673-4d4f5219ba72/
        prix_bas = 7500
        prix_haut = 52000
        data_paris = data_paris[(prix_bas<data_paris["prix_m²"]) & (data_paris["prix_m²"]<prix_haut)]
        data_paris.dropna(inplace=True)
        data_paris.sort_values(by=["Commune", "Voie", "Type de voie", "No voie"], inplace = True)
        
        # Outliers are now removed with the IQR method in each arrondissement separately
        c = 0
        for name in data_paris["Commune"].unique() :
            df_commune = data_paris[data_paris["Commune"]==name]
            Q1 = df_commune["prix_m²"].quantile(0.25)
            Q3 = df_commune["prix_m²"].quantile(0.75)
            IQR = Q3 - Q1    #IQR is interquartile range. 
            index_to_filter = df_commune[(df_commune["prix_m²"] < Q1 - 1.5 * IQR) | (df_commune["prix_m²"] > Q3 + 1.5 *IQR)].index
            c = c + len(index_to_filter)
            data_paris.drop(index_to_filter, inplace=True)
        
        print("Creation of the id column to merge with shapefile_df : parcelles.shp")
        data_paris["id"] = data_paris["Code departement"].astype(str) + data_paris["Code commune"].astype(str)+ "000" + data_paris["Section"].astype(str)
        data_paris["id"]=data_paris.apply(lambda row : add(row["id"], str(row['No plan'])), axis=1)
        data_paris["id"] = data_paris["id"].astype(str)
        shapefile_df["id"] = shapefile_df["id"].astype(str)
        data_paris = data_paris.merge(shapefile_df[["id","x", "y"]] , on="id", how='left')
        
        print("Creation of the final Dataframe for the model training")
        data_paris_knn = data_paris[["x", "y", "Commune", "prix_m²"]].dropna().copy()
        data_paris_knn.to_pickle("./data_paris_knn.pkl") # Backup fo the data_paris_knn Dataframe in pickle format.

        # Dataframes for the final training and prediction
        data_paris_knn_X = data_paris_knn[["x", "y"]]
        data_paris_knn_Y = data_paris_knn["prix_m²"]
        
        # Dataframes used for the StratifiedSplit : based on the column "Commune"
        # Only the index will be used with the previous Dataframes data_paris_knn_X & data_paris_knn_Y
        data_paris_knn_X_strat = data_paris_knn[["x", "y", "prix_m²"]]
        data_paris_knn_Y_strat = data_paris_knn["Commune"]
        
        print("Training KNeighborsRegressor model")
        # Use of a Stratified method with 10% of the training data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_index, test_index in sss.split(data_paris_knn_X_strat, data_paris_knn_Y_strat):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = data_paris_knn_X.iloc[train_index], data_paris_knn_X.iloc[test_index]
            y_train, y_test = data_paris_knn_Y.iloc[train_index], data_paris_knn_Y.iloc[test_index]
        
        # Hyperparameter k tested on a range from 1 to 300
        # The mean absolute percentage error is used as training 
        # Source : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
        k_list = list(np.arange(1, 300))
        liste_pourcentage = np.zeros(len(k_list))
        for k in k_list:
            neigh = KNeighborsRegressor(n_neighbors=k)
            neigh.fit(X_train, y_train)
            liste_pourcentage[k-1]=mean_absolute_percentage_error(y_test, neigh.predict(X_test))

        # Model saved with the best Hyperparameter
        neigh = KNeighborsRegressor(n_neighbors=liste_pourcentage.argmin()+1)
        neigh.fit(data_paris_knn_X, data_paris_knn_Y)
        filename = 'model.joblib'
        dump(neigh, filename) 

        print(f"KNeighborsRegressor model trained with k = {liste_pourcentage.argmin()+1}")
        print(f"End function Train --> call 'application_paris.py predict' for prediction")
        
    # -------------------------------------------------------------------------------
    if option == 'predict':
        """
        Function to obtain the prediction of the model.
        An API call is realized to transform, an adress in Paris, in its (x,y) coordinates (Lambert 93 projection)
        Source : https://geo.api.gouv.fr/adresse
        """
        print("Price/m² Prediction of the selected adress in Paris")
        adresse = input("Entrez adresse (ex : 6 Pl. du Colonel Bourgoin, 75012 Paris) : ")

        # Load model and data_paris_knn Dataframe to get neighbors back.
        neigh = load('model.joblib')  
        data_paris_knn = pd.read_pickle("./data_paris_knn.pkl")
        
        base_url = "https://api-adresse.data.gouv.fr/search/?"
        params = {'q' : adresse}
        response =  requests.get(base_url, params=params)
        data = response.json()
        x = data['features'][0]['properties']['x']
        y = data['features'][0]['properties']['y']
        
        # Neighbors retrieved + calculation of the prediction, min and max within neighbors
        knn_prediction = int(np.round(neigh.predict(np.array([[x, y]])),0)[0])
        knn_prediction_min = int(np.round(data_paris_knn.iloc[neigh.kneighbors(np.array([[x, y]]))[1][0]]['prix_m²'].min(),0))
        knn_prediction_max = int(np.round(data_paris_knn.iloc[neigh.kneighbors(np.array([[x, y]]))[1][0]]['prix_m²'].max(),0))

        # Neighbors retrieved + calculation of the confidence interval (instead of min max from neighbors)
        predict_voisins = data_paris_knn.iloc[neigh.kneighbors(np.array([[x, y]]))[1][0]]['prix_m²']
        knn_interval_confiance_inf = int(np.round(predict_voisins.mean() - 1.96*predict_voisins.std(),0))
        knn_interval_confiance_sup = int(np.round(predict_voisins.mean() + 1.96*predict_voisins.std(),0))

        
        print(f"Longitude x (Projection Lambert 93) : {x}")
        print(f"Latitude y (Projection Lambert 93) : {y}")
        print(f"Prédiction moyenne prix/m² : {knn_prediction} €")
        print(f"Prédiction prix/m² min: {knn_interval_confiance_inf} €")
        print(f"Prédiction prix/m² max: {knn_interval_confiance_sup} €")
        

if __name__ == "__main__":
    main()