import numpy as np
import pandas as pd
from data_processing import reader, analizer, write_to_csv
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import math

numerical_values = ['Duration_Years', 'Living_Cost_Index', 'Rent_USD', 'Visa_Fee_USD', 'Insurance_USD', 'Exchange_Rate']
categorical_values = ['Country', 'City', 'University', 'Program', 'Level']
target_attribute = 'Tuition_USD'

def matrix_analizer(columns, processor, column):
  if not hasattr(processor, 'is_fitted_'):
    processor.fit(columns)
  converted_num_values = processor.transform(columns)
  if sparse.issparse(converted_num_values):
    converted_num_values = converted_num_values.toarray()
  
  bias = np.hstack([np.ones((converted_num_values.shape[0], 1)), converted_num_values])
  culmn = column.to_numpy().reshape(-1, 1)
    
  return bias, culmn

def calulations(bias, culmn):
  try:
    θ = np.linalg.inv(bias.T @ bias) @ bias.T @ culmn
  except np.linalg.LinAlgError:
    θ = np.linalg.pinv(bias.T @ bias) @ bias.T @ culmn
  prediction = bias @ θ
  prediction = np.round(prediction, 2)

  return prediction

def main():
  file = r'C:\Users\37529\PythonProjects\Project2\International_Education_Costs.csv' 
  data_frame = reader(file)
  columns, column, processor = analizer(data_frame)

  train, temp = train_test_split(data_frame, test_size=0.2, random_state=42)
  validation, test = train_test_split(temp, test_size=0.25, random_state=42)
  list_of_sets = [('train', train), ('val', validation), ('test', test)]
    
  for name, en in list_of_sets:
    col = en.drop(columns=[target_attribute])
    targ = en[target_attribute]

    bias, culmn = matrix_analizer(col, processor, targ)  
    prediction = calulations(bias, culmn)
    output_file = f'my_ln_{name}'  
    write_to_csv(output_file, en['Country'], en['University'], prediction, targ)

    mean_abs_er = mean_absolute_error(targ, prediction)
    rout_mn_sqrt_er = math.sqrt(mean_squared_error(targ, prediction))
    mean_abs_er = round(mean_abs_er, 2)
    rout_mn_sqrt_er = round(rout_mn_sqrt_er, 2)
    
    print()
    print('Model: my ln, set: ', name)
    print('Mean average error: ', mean_abs_er)
    print('Rout mean squared error: ', rout_mn_sqrt_er)

if __name__ == '__main__':
    main()