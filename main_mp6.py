import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score


class UserPredictor:
    def __init__(self):
        self.model=Pipeline([
        ("pf",PolynomialFeatures(degree=2,include_bias=False)),
        ("std",StandardScaler()),
        ("lr",LogisticRegression(max_iter=500))
        ])
        self.xcols=['age','past_purchase_amt','total_duration','average_duration','max_duration']
    
    def fit(self,train_users,train_logs,train_y):
        train_x=self.preprocess(train_users,train_logs)
        train_x=train_x[self.xcols]
        self.model.fit(train_x,train_y['clicked'])
       
        scores = cross_val_score(self.model, train_x, train_y['clicked'], cv=5)
        
        print(f"AVG:{scores.mean()},STD:{scores.std()}")
        
        
    def predict(self,test_users,test_logs):
        test_x=self.preprocess(test_users,test_logs)
        test_x=test_x[self.xcols]
        predictions=self.model.predict(test_x)
        
        return predictions
        
    def preprocess(self,users_data,logs_data):
        
        merged_data=pd.merge(users_data,logs_data,on='id',how='left')
        summary_data=merged_data.groupby("id").agg({'duration':['sum','mean','max']})
        summary_data.columns=['total_duration','average_duration','max_duration']
        finished_data=pd.merge(users_data,summary_data,on='id',how='left')
        finished_data.fillna(0,inplace=True)
        finished_data=finished_data.select_dtypes(include=[np.number])
        
        return finished_data
        
# Example usage
if __name__ == "__main__":
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
