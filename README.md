# Description of the project

**I analyzed the International Education Costs dataset to predict tuition fees using machine learning. Linear regression overfit badly (train MAE: 1519 vs test MAE: 1.1M). Decision trees performed best (test MAE: 2679), while SVM was stable but less accurate. My custom gradient descent implementation avoided overfitting (test MAE: 16440) unlike the unstable closed-form solution. Decision trees proved most effective for this task.**

# Process of the analysis

**I processed the International Education Costs Dataset by separating categorical (country, university) and numerical features (tuition, costs). Through statistical analysis, I identified key patterns in education pricing across different regions.To ensure robustness, I applied k-fold cross-validation throughout the modeling phase. I also explored error distributions and plotted learning curves to assess model convergence and generalization. Regularization techniques (L1 and L2) were tested to mitigate overfitting in linear models. Data preprocessing included balancing the dataset (e.g., handling outliers and class imbalance), feature scaling, and encoding. I also performed hyperparameter optimization using grid search for decision trees and SVM. For modeling, I compared linear regression, decision trees, and SVM, evaluating performance via MAE/RMSE metrics. The analysis revealed significant variations in tuition costs and demonstrated that decision trees achieved the most accurate predictions, while linear regression showed severe overfitting issues. My custom gradient descent implementation provided stable results without the numerical instability of the closed-form solution.**

# How to run my code

**In order to run my code on another computer, the following steps are required:**

1. Clone the repository from GitHub (git clone https://github.com/V-tasya/Machine-learning-with-Python.git).
2. Create a virtual environment (python -m venv venv 
On Mac or Linux use: source venv/bin/activate  
On Windows use: venv\Scripts\activate).
3. Install dependences (pip install -r requirements.txt)
4. To se the results, run one of this files (data_processing, linear_regression.py, linear_regression_with_gradient.py)

# Notes

1. Ensure you have Python 3.13 and pip installed.