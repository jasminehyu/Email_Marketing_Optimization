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
   git clone https://github.com/yourusername/yourproject.git
