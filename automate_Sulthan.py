"""
Automated Data Preprocessing Script
=====================================
Script ini melakukan preprocessing data secara otomatis
berdasarkan eksperimen yang telah dilakukan pada notebook.

Author: [Nama Anda]
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import argparse
import os

warnings.filterwarnings('ignore')


class WineQualityPreprocessor:
    """
    Class untuk melakukan preprocessing pada Wine Quality Dataset
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        test_size : float
            Proporsi data untuk test set (default: 0.2)
        random_state : int
            Random state untuk reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, filepath):
        """
        Load dataset dari file
        
        Parameters:
        -----------
        filepath : str
            Path ke file dataset (CSV)
            
        Returns:
        --------
        pd.DataFrame
            Dataset yang telah dimuat
        """
        print(f"📁 Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully! Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """
        Menangani missing values dengan median imputation
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset input
            
        Returns:
        --------
        pd.DataFrame
            Dataset tanpa missing values
        """
        print("\n🔧 Handling missing values...")
        missing_before = df.isnull().sum().sum()
        print(f"   Missing values before: {missing_before}")
        
        if missing_before > 0:
            df[df.columns] = self.imputer.fit_transform(df)
            
        missing_after = df.isnull().sum().sum()
        print(f"   Missing values after: {missing_after}")
        return df
    
    def remove_duplicates(self, df):
        """
        Menghapus data duplikat
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset input
            
        Returns:
        --------
        pd.DataFrame
            Dataset tanpa duplikat
        """
        print("\n🔧 Removing duplicate data...")
        duplicates_before = df.duplicated().sum()
        print(f"   Duplicates before: {duplicates_before}")
        
        df = df.drop_duplicates()
        
        duplicates_after = df.duplicated().sum()
        print(f"   Duplicates after: {duplicates_after}")
        print(f"   Rows after removing duplicates: {len(df)}")
        return df
    
    def feature_engineering(self, df):
        """
        Membuat fitur baru dan encoding target variable
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset input
            
        Returns:
        --------
        pd.DataFrame
            Dataset dengan fitur baru
        """
        print("\n🔧 Feature engineering...")
        
        # Categorize quality
        def categorize_quality(quality):
            if quality <= 5:
                return 'Low'
            elif quality <= 6:
                return 'Medium'
            else:
                return 'High'
        
        df['quality_category'] = df['quality'].apply(categorize_quality)
        
        # Encoding
        quality_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df['quality_encoded'] = df['quality_category'].map(quality_mapping)
        
        print("   Quality distribution:")
        print(df['quality_category'].value_counts().to_dict())
        
        return df
    
    def remove_outliers_iqr(self, df, column):
        """
        Menghapus outliers menggunakan IQR method
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset input
        column : str
            Nama kolom untuk remove outliers
            
        Returns:
        --------
        pd.DataFrame
            Dataset tanpa outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def handle_outliers(self, df):
        """
        Menangani outliers pada semua fitur numerik
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset input
            
        Returns:
        --------
        pd.DataFrame
            Dataset tanpa outliers
        """
        print("\n🔧 Handling outliers...")
        rows_before = len(df)
        
        features_to_clean = ['fixed acidity', 'volatile acidity', 'citric acid', 
                             'residual sugar', 'chlorides', 'free sulfur dioxide',
                             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        
        for feature in features_to_clean:
            if feature in df.columns:
                df = self.remove_outliers_iqr(df, feature)
        
        rows_after = len(df)
        print(f"   Rows before: {rows_before}")
        print(f"   Rows after: {rows_after}")
        print(f"   Outliers removed: {rows_before - rows_after}")
        
        return df
    
    def scale_features(self, X_train, X_test):
        """
        Standardisasi fitur menggunakan StandardScaler
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Testing features
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_test_scaled)
        """
        print("\n🔧 Scaling features...")
        
        # Fit scaler pada train data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # Transform test data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"   Train features scaled: {X_train_scaled.shape}")
        print(f"   Test features scaled: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, df):
        """
        Split data menjadi train dan test set
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset yang sudah diproses
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        print("\n🔧 Splitting data...")
        
        # Separate features and target
        X = df.drop(['quality', 'quality_category', 'quality_encoded'], axis=1)
        y = df['quality_encoded']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"   X_train: {X_train.shape}")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir='preprocessed_data'):
        """
        Menyimpan data yang sudah diproses ke file
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : pd.DataFrame/pd.Series
            Data yang akan disimpan
        output_dir : str
            Directory untuk menyimpan output
        """
        print(f"\n💾 Saving preprocessed data to '{output_dir}'...")
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine and save
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(f'{output_dir}/wine_quality_train.csv', index=False)
        test_data.to_csv(f'{output_dir}/wine_quality_test.csv', index=False)
        
        # Save full preprocessed data
        full_data = pd.concat([train_data, test_data], axis=0)
        full_data.to_csv(f'{output_dir}/wine_quality_preprocessed.csv', index=False)
        
        print(f"   ✓ Train data saved: {output_dir}/wine_quality_train.csv")
        print(f"   ✓ Test data saved: {output_dir}/wine_quality_test.csv")
        print(f"   ✓ Full data saved: {output_dir}/wine_quality_preprocessed.csv")
    
    def preprocess_pipeline(self, input_filepath, output_dir='preprocessed_data'):
        """
        Pipeline lengkap untuk preprocessing data
        
        Parameters:
        -----------
        input_filepath : str
            Path ke raw data
        output_dir : str
            Directory untuk output
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        print("="*70)
        print("🚀 AUTOMATED DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # 1. Load data
        df = self.load_data(input_filepath)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df)
        
        # 3. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 4. Feature engineering
        df = self.feature_engineering(df)
        
        # 5. Handle outliers
        df = self.handle_outliers(df)
        
        # 6. Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # 7. Scale features
        X_train, X_test = self.scale_features(X_train, X_test)
        
        # 8. Save preprocessed data
        self.save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)
        
        print("\n" + "="*70)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return X_train, X_test, y_train, y_test


def main():
    """
    Main function untuk menjalankan preprocessing
    """
    parser = argparse.ArgumentParser(description='Automated Wine Quality Data Preprocessing')
    parser.add_argument('--input', type=str, default='wine_quality_raw.csv',
                        help='Path to raw data file')
    parser.add_argument('--output', type=str, default='preprocessed_data',
                        help='Output directory for preprocessed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = WineQualityPreprocessor(
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        input_filepath=args.input,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
