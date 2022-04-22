from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle
import random


app = Flask(__name__)
api = Api(app)

filepath = 'C:/Users/micha/OneDrive/LHL/_DataScienciBootCamp/w07/d4/mini_project_4/'

def log_transform(x):
    return np.log(x.astype(float) + 1)

class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        with open(filepath+'bestmodel.pickle', 'rb') as f:
            best_model = pickle.load(f)
        res = best_model.predict_proba(df)
        return res.tolist()
    
api.add_resource(Scoring, '/scoring')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)   
