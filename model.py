
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


#loading the dataset
data = load_iris()

def train():
    """
    Trains a linear regression model on a flower class dataset.
    Params: None
    output: LogisticRegression instance
    """

    #splitting the dataset and training
    x = data['data']
    y = data['target']


    

    #fitting the model to the data

    regression = LogisticRegression()
    regression.fit(x,y)

    return regression


def deploy(input):
    """
    Deployment of the model.
    Params: Input house features (array-like)
    Outputs: Estimated inferred value (float or list-of-floats)
    """

    linear_model = train()

    if not isinstance(input, (np.ndarray,list, pd.DataFrame)):
        input = np.array(input).reshape(-1,13)
    
    
    output = linear_model.predict(input)


    return str(data['target_names'][output])
