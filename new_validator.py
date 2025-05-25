from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math
from scipy import sparse
from data_processing import reader, analizer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Validator:

    def __init__(self, processor, target_attribute):
        self.processor = processor
        self.target_attribute = target_attribute
        self.splits = 3
        self.category_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.numeric_processor = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        self.oversampler = SMOTE(random_state=42)
        self.undersampler = RandomUnderSampler(random_state=42)

    def prepare_data(self, data_frame):
        data_frame['Tuition_Class'] = self.numeric_processor.fit_transform(
            data_frame[[self.target_attribute]]).astype(int)
        categorical_values = ['Country', 'City', 'University', 'Program', 'Level']
        features = data_frame[categorical_values]
        target = data_frame['Tuition_Class']
        features_encoded = self.category_processor.fit_transform(features)
        
        return features_encoded, target

    def evaluate_approaches(self, features, target):
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.3, random_state=42, stratify=target)
        self._train_and_evaluate(features_train, target_train, features_test, target_test, "Original data")
        features_oversampled, target_oversampled = self.oversampler.fit_resample(
            features_train, target_train)
        self._train_and_evaluate(features_oversampled, target_oversampled, features_test, target_test, "SMOTE")
        
        features_undersampled, target_undersampled = self.undersampler.fit_resample(
            features_train, target_train)
        self._train_and_evaluate(features_undersampled, target_undersampled, features_test, target_test, "Undersampling")

    def _train_and_evaluate(self, train_features, train_target, test_features, test_target, approach_name):
        model = LinearRegression()
        model.fit(train_features, train_target)
        predictions = model.predict(test_features)
        
        mae = mean_absolute_error(test_target, predictions)
        mse = mean_squared_error(test_target, predictions)
        
        print(f"\nName: {approach_name}")
        print(f"MAE: {round(mae, 2)}")
        print(f"RMSE: {round(mse, 2)}")

    def cross_validate(self, data_frame):
        features, target = self.prepare_data(data_frame)
        kf = KFold(n_splits=self.splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(features), 1):
            print(f"\n{fold} set")
            train_features = features[train_idx]
            train_target = target.iloc[train_idx]
            test_features = features[test_idx]
            test_target = target.iloc[test_idx]
            self._train_and_evaluate(train_features, train_target, test_features, test_target, "Original data")

def main():
    file_path = 'International_Education_Costs.csv'
    data_frame = reader(file_path)
    
    validator = Validator(processor=None, target_attribute='Tuition_USD')
    
    print("Cross-Validation Results")
    validator.cross_validate(data_frame)
    
    print("\nStandard Evaluation with Balancing Methods")
    features, target = validator.prepare_data(data_frame)
    validator.evaluate_approaches(features, target)

if __name__ == '__main__':
    main()