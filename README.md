# Smart Marketing for TV Sales

## Overview
This project aims to optimize email marketing for TV promotions on a retail website. By analyzing past user interactions with similar promotional emails, we can predict which users are likely to be interested in upcoming promotions, thus minimizing unwanted emails and improving user experience.

## Features
- Predict user interest based on historical data.
- Utilizes logistic regression with polynomial features and standard scaling.
- Provides cross-validation for accuracy estimation.

## Dataset
The dataset consists of user information, webpage visit logs, and email click data. These datasets are used to train and evaluate the predictive model.

1. `users.csv`: Information about each user.
2. `logs.csv`: Information about the webpages visited by each user.
3. `clicked.csv`: Clicked=1 means the user clicked the email, Clicked=0 means they did not.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/jasminehyu/Project_ML_sale.git
2. Navigate to the project directory:
   ```bash
   cd Project_ML_sale
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the classifier:
   ```bash
   python main.py
## Example
import pandas as pd
from main_mp6 import UserPredictor

# Load your data
train_users = pd.read_csv('path_to_train_users.csv')
train_logs = pd.read_csv('path_to_train_logs.csv')
train_clicked = pd.read_csv('path_to_train_clicked.csv')

# Initialize and fit the model
predictor = UserPredictor()
predictor.fit(train_users, train_logs, train_clicked)

# Make predictions
test_users = pd.read_csv('path_to_test_users.csv')
test_logs = pd.read_csv('path_to_test_logs.csv')
predictions = predictor.predict(test_users, test_logs)
print(predictions)

##Contact
If you have any questions, please contact Jasmine Yu at jasmineyuhhy@gmail.com.
