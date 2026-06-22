"""
Machine Learning models for LBW decision
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os


class LBWMLModels:
    """Ensemble ML models for LBW decisions"""
    
    def __init__(self, random_state=42):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = None
        self.imputer = None
        self.feature_cols = None
        self.random_state = random_state
    
    def train(self, data, label_col='Label', test_size=0.2):
        """
        Train models on data
        
        Args:
            data: DataFrame with features and labels
            label_col: Column name for labels
            test_size: Proportion for test set
            
        Returns:
            Dict with training results
        """
        # Prepare data
        self.feature_cols = [c for c in data.columns if c not in ['Label', 'Video_Name', 'Calibration_Method']]
        
        X = data[self.feature_cols]
        y = data[label_col]
        
        # Impute and scale
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        X_imp = pd.DataFrame(self.imputer.fit_transform(X), columns=self.feature_cols)
        X_sc = pd.DataFrame(self.scaler.fit_transform(X_imp), columns=self.feature_cols)
        
        # Train-test split
        Xtr, Xte, ytr, yte = train_test_split(
            X_sc, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Apply SMOTE for class balance
        if ytr.value_counts().min() >= 5:
            smote = SMOTE(random_state=self.random_state, 
                         k_neighbors=min(5, ytr.value_counts().min() - 1))
            Xtr_bal, ytr_bal = smote.fit_resample(Xtr, ytr)
            print(f"✅ After SMOTE: {dict(pd.Series(ytr_bal).value_counts())}")
        else:
            Xtr_bal, ytr_bal = Xtr, ytr
        
        # Train Random Forest
        print("\n🌲 Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=self.random_state, 
            n_jobs=-1, class_weight='balanced'
        )
        self.rf_model.fit(Xtr_bal, ytr_bal)
        
        # Train XGBoost
        print("🚀 Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=self.random_state, eval_metric='logloss'
        )
        self.xgb_model.fit(Xtr_bal, ytr_bal)
        
        # Evaluate
        rf_pred = self.rf_model.predict(Xte)
        xgb_pred = self.xgb_model.predict(Xte)
        
        results = {
            'rf_accuracy': accuracy_score(yte, rf_pred),
            'rf_f1': f1_score(yte, rf_pred),
            'xgb_accuracy': accuracy_score(yte, xgb_pred),
            'xgb_f1': f1_score(yte, xgb_pred),
            'rf_confusion': confusion_matrix(yte, rf_pred),
            'xgb_confusion': confusion_matrix(yte, xgb_pred),
        }
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   RF  Acc={results['rf_accuracy']:.3f} F1={results['rf_f1']:.3f}")
        print(f"   XGB Acc={results['xgb_accuracy']:.3f} F1={results['xgb_f1']:.3f}")
        
        return results
    
    def predict(self, features):
        """
        Make prediction on new features
        
        Args:
            features: Dict or DataFrame with features
            
        Returns:
            Tuple (out_prob, out_decision, rf_prob, xgb_prob)
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Prepare input
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        features_df = features_df[self.feature_cols]
        features_imp = self.imputer.transform(features_df)
        features_sc = self.scaler.transform(features_imp)
        
        # Predict
        rf_probs = self.rf_model.predict_proba(features_sc)
        xgb_probs = self.xgb_model.predict_proba(features_sc)
        
        rf_pred = self.rf_model.predict(features_sc)
        xgb_pred = self.xgb_model.predict(features_sc)
        
        # Ensemble
        avg_out_prob = (rf_probs[:, 1] + xgb_probs[:, 1]) / 2
        ensemble_pred = (avg_out_prob > 0.5).astype(int)
        
        return {
            'rf_prob': rf_probs[0, 1],
            'xgb_prob': xgb_probs[0, 1],
            'rf_pred': rf_pred[0],
            'xgb_pred': xgb_pred[0],
            'ensemble_pred': ensemble_pred[0],
            'ensemble_prob': avg_out_prob[0],
            'ensemble_confidence': max(avg_out_prob[0], 1 - avg_out_prob[0]) * 100
        }
    
    def save(self, output_dir):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.rf_model, f"{output_dir}/rf_model.pkl")
        joblib.dump(self.xgb_model, f"{output_dir}/xgb_model.pkl")
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.imputer, f"{output_dir}/imputer.pkl")
        joblib.dump(self.feature_cols, f"{output_dir}/feature_cols.pkl")
    
    def load(self, model_dir):
        """Load trained models"""
        self.rf_model = joblib.load(f"{model_dir}/rf_model.pkl")
        self.xgb_model = joblib.load(f"{model_dir}/xgb_model.pkl")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.imputer = joblib.load(f"{model_dir}/imputer.pkl")
        self.feature_cols = joblib.load(f"{model_dir}/feature_cols.pkl")