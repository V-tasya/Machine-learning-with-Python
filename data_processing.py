import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

numerical_values = ['Duration_Years', 'Living_Cost_Index', 'Rent_USD', 'Visa_Fee_USD', 'Insurance_USD', 'Exchange_Rate']
categorical_values = ['Country', 'City', 'University', 'Program', 'Level']
target_attribute = 'Tuition_USD'
models = {'ln': LinearRegression(), 'dt': DecisionTreeRegressor(random_state=42), 'svr': SVR()}
result = {}


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
  #transformed_columns = preprocessing.fit_transform(columns_without_target)
  #print('Form of the processed data: ', transformed_columns.shape)

  return columns_without_target, target_variables_column, preprocessing

def training(columns_without_target, target_variables_column, preprocessing):
  columns_without_target_train, columns_without_target_test, target_variables_column_train, target_variables_column_test = train_test_split(columns_without_target, target_variables_column, test_size=0.2, random_state=42)
  
  for id, model in models.items():
    trainer = Pipeline([('processor', preprocessing), ('model', model)])
    trainer.fit(columns_without_target_train, target_variables_column_train)
    prediction = trainer.predict(columns_without_target_test)
    mean_abs_er = mean_absolute_error(target_variables_column_test, prediction)
    rout_mn_sqrt_er = math.sqrt(mean_squared_error(target_variables_column_test, prediction))

    print('Model: ', id)
    print('Mean average error: ', mean_abs_er)
    print('Rout mean squared error: ', rout_mn_sqrt_er)

def main():
  file = r'C:\Users\37529\PythonProjects\Project2\International_Education_Costs.csv'
  df = reader(file)
  columns, column, processor = analizer(df)
  training(columns, column, processor)

if __name__ == '__main__':
  main()