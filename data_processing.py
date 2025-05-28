import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import math
import csv
import numpy as np

numerical_values = ['Duration_Years', 'Living_Cost_Index', 'Rent_USD', 'Visa_Fee_USD', 'Insurance_USD', 'Exchange_Rate']
categorical_values = ['Country', 'City', 'University', 'Program', 'Level']
target_attribute = 'Tuition_USD'
models = {'ln': LinearRegression(), 
          #'dd': DecisionTreeRegressor(random_state=42), 
          #'svr': SVR(),
          'r': Ridge(alpha=1.0),   
          'l': Lasso(alpha=0.1)}
result = {} 

def reader(file):
  try:
    data_frame = pd.read_csv(file)
    if data_frame.empty:
      print("The file is empty.")
      return 
    print("rows, columns:", data_frame.shape) 
  except FileNotFoundError:
    print("The file wasn't found.")
    return
  except Exception as exeption:
    print(f"Error: {exeption}")
    return
  
  return data_frame

def analizer(data_frame):
  columns_without_target = data_frame.drop(columns=[target_attribute])
  target_variables_column = data_frame[target_attribute]

  num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                    #('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                    ('scaler', StandardScaler())])
  cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
  common_transformer = ColumnTransformer(transformers=[('num', num_transformer, numerical_values), ('cat', cat_transformer, categorical_values)])

  preprocessing = Pipeline(steps=[('preprocessor', common_transformer)])
  #transformed_columns = preprocessing.fit_transform(columns_without_target)
  #print('Form of the processed data: ', transformed_columns.shape)

  return columns_without_target, target_variables_column, preprocessing

def training(columns_without_target, target_variables_column, preprocessing, countries, universities):
  columns_without_target_temp, columns_without_target_test, target_variables_column_temp, target_variables_column_test = train_test_split(columns_without_target, target_variables_column, test_size=0.2, random_state=42, shuffle=False)
  columns_without_target_train, columns_without_target_val, target_variables_column_train, target_variables_column_val = train_test_split(columns_without_target_temp, target_variables_column_temp, test_size=0.25, random_state=42, shuffle=False)

  countries_train = countries.loc[columns_without_target_train.index]
  countries_val = countries.loc[columns_without_target_val.index]
  countries_test = countries.loc[columns_without_target_test.index]
  university_train = universities.loc[columns_without_target_train.index]
  university_val = universities.loc[columns_without_target_val.index]
  university_test = universities.loc[columns_without_target_test.index]

  list_of_sets = [('train', columns_without_target_train, target_variables_column_train, countries_train, university_train),
                  ('validate', columns_without_target_val, target_variables_column_val, countries_val, university_val),
                  ('test', columns_without_target_test, target_variables_column_test, countries_test, university_test)]
  
  for id, model in models.items():
    trainer = Pipeline([('processor', preprocessing), ('model', model)])
    trainer.fit(columns_without_target_train, target_variables_column_train)
    for name, col, targ, countr, uni in list_of_sets:
      prediction = trainer.predict(col)
      prediction = np.round(prediction, 2)
      output_file = f'{id}_{name}'
      write_to_csv(output_file, countr, uni, prediction, targ)
    
      mean_abs_er = mean_absolute_error(targ, prediction)
      rout_mn_sqrt_er = math.sqrt(mean_squared_error(targ, prediction))
      mean_abs_er = round(mean_abs_er, 2)
      rout_mn_sqrt_er = round(rout_mn_sqrt_er, 2)
    
      print()
      print('Model: ', id, ', set: ', name)
      print('Mean average error: ', mean_abs_er)
      print('Rout mean squared error: ', rout_mn_sqrt_er)
      if hasattr(model, 'coef_'):
        print("First 16 weights :")
        coefs = model.coef_
        print(coefs[:16])

def write_to_csv(file,countries, universities, prediction, enum):
  data_frame = pd.DataFrame({
    'Country': countries,
    'University': universities,
    'Predicted_Tuition_USD': prediction.flatten(),
    'Tuition_USD': enum.values
  })
  data_frame.to_csv(f'{file}_predictions.csv', index=False)

def main():
  file = r'C:\Users\37529\PythonProjects\Project2\International_Education_Costs.csv'
  df = reader(file)
  columns, column, processor = analizer(df)
  countries = columns['Country']
  universities = columns['University']
  training(columns, column, processor, countries, universities)

if __name__ == '__main__':
  main()