from typing import Any, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from kindel.models.base import BaseModel


class CustomModel(BaseModel):
    def __init__(self, wandb_logger=None, **hyperparameters):
        super().__init__()
        self.wandb_logger = wandb_logger
        # TODO: Initialize your model and hyperparameters
        self.model = None
        
    def prepare_dataset(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame) -> Any:
        """
        Convert DataFrames into the format needed for your model.
        Access to both molecule SMILES and synthon SMILES (smiles_a, smiles_b, smiles_c)
        """
        # Example: Calculate enrichment scores for training data
        def calculate_enrichment(row):
            target_counts = np.mean([row[f'seq_target_{i}'] for i in range(1, 4)])
            matrix_counts = np.mean([row[f'seq_matrix_{i}'] for i in range(1, 4)])
            return np.log2((target_counts + 1) / (matrix_counts + 1))
            
        train_y = df_train.apply(calculate_enrichment, axis=1).values
        
        # TODO: Convert SMILES to your desired molecular representation
        train_x = self.featurize(df_train)
        valid_x = self.featurize(df_valid)
        test_x = self.featurize(df_test)
        
        # Return in format your model expects
        return type('Dataset', (), {
            'train': type('Split', (), {'x': train_x, 'y': train_y})(),
            'valid': type('Split', (), {'x': valid_x, 'y': None})(),
            'test': type('Split', (), {'x': test_x, 'y': None})(),
        })()

    def featurize(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert SMILES strings to molecular features"""
        # TODO: Implement your featurization logic
        # Example: Simple Morgan fingerprints
        fps = []
        for smiles in df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            fps.append(np.array(fp))
        return np.array(fps)

    def train(self):
        """Train your model"""
        # TODO: Implement your training logic
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for new data"""
        # TODO: Implement prediction logic
        pass