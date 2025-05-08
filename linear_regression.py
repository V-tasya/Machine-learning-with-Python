import numpy as np
import pandas as pd
from data_processing import reader, analizer
from scipy import sparse

numerical_values = ['Duration_Years', 'Living_Cost_Index', 'Rent_USD', 'Visa_Fee_USD', 'Insurance_USD', 'Exchange_Rate']

def matrix_analizer(columns, processor, column):
    converted_num_values = processor.fit_transform(columns)
    if sparse.issparse(converted_num_values):
        converted_num_values = converted_num_values.toarray()
  
    #print(f"{converted_num_values.shape}")
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

def write_to_csv(prediction, countries, universities, file):
    pd.DataFrame({
        'Country': countries,
        'University': universities,
        'Predicted_Tuition_USD': prediction.flatten()
    }).to_csv(file, index=False)

def main():
    file = r'C:\Users\37529\PythonProjects\Project2\International_Education_Costs.csv'
    file_for_predictions = r'C:\Users\37529\PythonProjects\Project2\My_LinearRegression_Predictions.csv'
    
    df = reader(file)
    columns, column, processor = analizer(df)
    
    bias, culmn = matrix_analizer(columns, processor, column)
    prediction = calulations(bias, culmn)
    
    countries = columns['Country'].values
    universities = columns['University'].values
    write_to_csv(prediction, countries, universities, file_for_predictions)

if __name__ == '__main__':
    main()