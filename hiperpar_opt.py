from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import math
import pandas as pd
from data_processing import reader, analizer


def evaluate_model(model, features, target):
    prediction = model.predict(features)
    prediction = np.round(prediction, 2)
    mae = round(mean_absolute_error(target, prediction), 2)
    rmse = round(math.sqrt(mean_squared_error(target, prediction)), 2)
    return mae, rmse


def training(columns_without_target, target_variables_column, preprocessing, countries, universities):
    # Разделение данных
    columns_without_target_temp, columns_without_target_test, target_variables_column_temp, target_variables_column_test = train_test_split(
        columns_without_target, target_variables_column, test_size=0.2, random_state=42, shuffle=False)
    columns_without_target_train, columns_without_target_val, target_variables_column_train, target_variables_column_val = train_test_split(
        columns_without_target_temp, target_variables_column_temp, test_size=0.25, random_state=42, shuffle=False)

    # Подготовка дополнительных данных
    countries_train = countries.loc[columns_without_target_train.index]
    countries_val = countries.loc[columns_without_target_val.index]
    countries_test = countries.loc[columns_without_target_test.index]
    university_train = universities.loc[columns_without_target_train.index]
    university_val = universities.loc[columns_without_target_val.index]
    university_test = universities.loc[columns_without_target_test.index]

    list_of_sets = [
        ('train', columns_without_target_train, target_variables_column_train),
        ('validate', columns_without_target_val, target_variables_column_val),
        ('test', columns_without_target_test, target_variables_column_test)
    ]

    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42)
    }

    for model_id, model in models.items():
        print(model_id)
        print('Before optimization:')
        trainer = Pipeline([('processor', preprocessing), ('model', model)])
        trainer.fit(columns_without_target_train, target_variables_column_train)

        for name, col, targ in list_of_sets:
            mae, rmse = evaluate_model(trainer, col, targ)
            print(f'{name}: MAE={mae}, RMSE={rmse}')

        if isinstance(model, DecisionTreeRegressor):
            param_grid = {
                'model__max_depth': [3, 5, 7, 10],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(model, RandomForestRegressor):
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10, 15],
                'model__max_features': ['sqrt', 'log2', 0.5]
            }

        grid = GridSearchCV(
            Pipeline([('processor', preprocessing), ('model', model)]),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(columns_without_target_train, target_variables_column_train)
        best_model = grid.best_estimator_

        print("\nBest parameters:", grid.best_params_)
        print("Best RMSE:", round((-grid.best_score_)**0.5, 2))

        print('\nAfter optimization:')
        for name, col, targ in list_of_sets:
            mae, rmse = evaluate_model(best_model, col, targ)
            print(f'{name}: MAE={mae}, RMSE={rmse}')


def main():
    file = r'C:\Users\37529\PythonProjects\Project2\International_Education_Costs.csv'
    df = reader(file)
    columns, column, processor = analizer(df)

    countries = columns['Country']
    universities = columns['University']

    training(columns, column, processor, countries, universities)


if __name__ == '__main__':
    main()