#!/usr/bin/env python3
"""
Script para generar datos sintéticos de usuarios para targeting de promociones.
Este script simula un dataset realista para el problema de identificar qué usuarios
deben recibir promociones basado en su comportamiento transaccional y perfil.
"""

import os

import pandas as pd
import kagglehub
import shutil

class GetData:
    def __init__(self, kaggle_dataset_id="rajyellow46/wine-quality", download_path="../data"):
        self.kaggle_dataset_id = kaggle_dataset_id
        self.download_path = download_path
        self.column_mapping = mapeo_columnas = {
                                                'fixed acidity': 'fixed_acidity', 
                                                'volatile acidity': 'volatile_acidity',
                                                'citric acid': 'citric_acid', 
                                                'residual sugar': 'residual_sugar',
                                                'free sulfur dioxide': 'free_sulfur_dioxide',
                                                'total sulfur dioxide': 'total_sulfur_dioxide'
                                            }

    def download_data(self):
        """
        Descarga los datos de Kaggle.
        
        Args:
            kaggle_dataset_id (str): ID del dataset de Kaggle
            download_path (str): Ruta donde se descargarán los datos

        Returns:
            None
        """
        path = kagglehub.dataset_download(self.kaggle_dataset_id)

        # Definir carpeta destino en local (ej: datasets/glass dentro del proyecto)
        dest_path = os.path.join(os.getcwd(), self.download_path, "")

        # Crear carpeta si no existe
        os.makedirs(dest_path, exist_ok=True)

        # Copiar archivos descargados a la carpeta destino
        for file in os.listdir(path):
            shutil.copy(os.path.join(path, file), dest_path)

    def create_dataset(self):
        """Función principal para generar y guardar los datos."""

        df_wine_quality = pd.read_csv(self.download_path + "/winequalityN.csv")
        df_wine_quality = df_wine_quality.rename(columns=self.column_mapping)

        return df_wine_quality

