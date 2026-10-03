import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# # Nettoyer les données et sauvegarder df_clean
# df_clean = pd.read_csv('https://raw.githubusercontent.com/hajjajihamza/brief-3/refs/heads/main/data/row/dataset.csv')
# df_clean = df_clean.dropna(axis=0)
# df_clean = df_clean.drop(columns=['CUST_ID']).reset_index(drop=True)
# df_clean.to_csv('data/processed/df_clean.csv', index=False)
#
# columns = df_clean.columns.tolist()
#
# # Transformation logarithmique
# df_prepare_clustering = df_clean.copy()
#
# for column in columns:
#     df_prepare_clustering[column] = np.log1p(df_prepare_clustering[column])
#
# # Transformation StandardScaler
# scaler = StandardScaler()
# df_prepare_clustering[columns] = scaler.fit_transform(df_prepare_clustering[columns])
# print(df_prepare_clustering)

def cleaning_dataset(path_load:str, path_sae:str):
    # loader dataset
    df = pd.read_csv(path_load)

    # supprimr les line due quntine des value aberrantes
    df = df.dropna(axis=0)

    # suprimer les colonnes inutiles
    df = df.drop(columns=['CUST_ID']).reset_index(drop=True)

    #  sauvegarder le data frame cleaning
    # df.to_csv(path_sae, index=False)
    return df


def transform_logarithmique(df: pd.DataFrame, columns:list):
    for column in columns:
        df[column] = np.log1p(df[column])

    return df

def transform_standard_scaler(df:pd.DataFrame, columns:list):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df