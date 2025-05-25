from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math
from scipy import sparse
from linear_regression_with_gradient import matrix_analizer, calulations, target_attribute
from data_processing import reader, analizer

class Validator:

  def __init__(self, calculations, processor, target_attribute):
    self.calculations = calculations
    self.processor = processor
    self.target_attribute = target_attribute
    self.splits = 3
    self.rate = 0.01
    self.iterations = 100
    self.batch_size = 32

  def analizzer(self, columns, column):
    converted_num_values = self.processor.transform(columns)
    if sparse.issparse(converted_num_values):
        converted_num_values = converted_num_values.toarray()

    bias = np.hstack([np.ones((converted_num_values.shape[0], 1)), converted_num_values])
    culmn = column.to_numpy().reshape(-1, 1)
    return bias, culmn
  
  def start_validation(self, frame):
    columns_without_target = frame.drop(columns=[self.target_attribute])
    target_column = frame[self.target_attribute]
    sets = KFold(n_splits=self.splits, shuffle=True, random_state=42)
    set_num = 1

    for train_ind, test_ind in sets.split(columns_without_target):
      train_columns_without_target = columns_without_target.iloc[train_ind]
      test_columns_without_target = columns_without_target.iloc[test_ind]
      train_target_column = target_column.iloc[train_ind]
      test_target_column = target_column.iloc[test_ind]
      train_bias, train_column = self.analizzer(train_columns_without_target, train_target_column)
      test_bias, test_column = self.analizzer(test_columns_without_target, test_target_column)

      training_prediction = self.calculations(train_bias, train_column, self.rate, self.iterations, self.batch_size)
      θ = np.linalg.pinv(train_bias) @ train_column
      testing_prediction = test_bias @ θ
      testing_prediction = np.round(testing_prediction, 2)
      mean_abs_er = mean_absolute_error(test_target_column, testing_prediction)
      rout_mn_sqrt_er = math.sqrt(mean_squared_error(test_target_column, testing_prediction))

      print()
      print('For ', set_num, ' set:')
      print('Mean average error: ', round(mean_abs_er, 2))
      print('Rout mean squared error: ', round(rout_mn_sqrt_er, 2))
      set_num += 1

def main():
  file = r'C:\Users\37529\PythonProjects\Project2\International_Education_Costs.csv'
  frame = reader(file)
  _, _, processor = analizer(frame)
  cols = frame.drop(columns=[target_attribute])
  processor.fit(cols)

  validator = Validator(calulations,processor,target_attribute)
  validator.start_validation(frame)

if __name__ == '__main__':
  main()