import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numerical_values = ['Duration_Years', 'Living_Cost_Index', 'Rent_USD', 'Visa_Fee_USD', 'Insurance_USD', 'Exchange_Rate']
categorical_values = ['Country', 'City', 'University', 'Program', 'Level']
target_attribute = 'Tuition_USD'

def reader(file):
  data_frame = pd.read_csv(file)
  return data_frame

def analizer(data_frame):
  columns_without_target = data_frame.drop(columns=[target_attribute])
  target_variables_column = data_frame[target_attribute]

  num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
  cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
  common_transformer = ColumnTransformer(transformers=[('num', num_transformer, numerical_values), ('cat', cat_transformer, categorical_values)])

  preprocessing = Pipeline(steps=[('preprocessor', common_transformer)])
  transformed_columns = preprocessing.fit_transform(columns_without_target)
  print('Form of the processed data: ', transformed_columns.shape)

  return transformed_columns, target_variables_column

def main():
  file = r'C:\Users\37529\PythonProjects\Project2\International_Education_Costs.csv'
  df = reader(file)
  analizer(df)

if __name__ == '__main__':
  main()