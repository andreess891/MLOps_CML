import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, df):
        self.df = df
    
    def create_features(self):
        datos_white = self.df[self.df['type'] == 'white'] # Solo vino blanco
        datos_white = datos_white.drop(columns=['type']) # Eliminar columna redundante

        return datos_white