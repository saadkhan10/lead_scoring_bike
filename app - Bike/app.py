
from flask import Flask, render_template, request, redirect, url_for
import pickle
import statsmodels.api as sm
import pandas as pd
from xgboost import XGBClassifier
import sys
import logging

def predict_results(data):

    
    modelfile = open('bikemodel1.pkl', 'rb')
    input2 = pd.read_csv(data)
    input2["year"] = input2["year"].astype("object") 
    res = pickle.load(modelfile)
    input1 = input2.drop(["Lead Number"],axis = 1)
    input = pd.get_dummies(input1,columns = list(input1.select_dtypes(include=['object']).columns))

    prob = res.predict_proba(input)[:,1]
    pred_df1 = pd.DataFrame({"Lead Score":prob})
    pred_df1['Lead Score'] = round(pred_df1['Lead Score'] * 100, 2)
    pred_df = pd.concat([pred_df1, input2], axis=1)
    pred_df = pred_df.sort_values(by = ["Lead Score"], ascending = False )
    pred_df['Lead Score'] = pred_df['Lead Score'].astype(str)
    pred_df['Lead Score'] = pred_df['Lead Score'] + '%'
    pred_df.rename(columns={'Total Time Spent on Website':'Session Time'}, inplace=True)
    
    return pred_df
    


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['data_file']
    if data.filename != '':
        df = predict_results(data)

        return render_template('result.html', tables=[df.to_html(index = False, classes = "table thead-dark",justify = "center")], titles=['na', 'Lead Scores'])

    return "Error occurred."


if __name__ == '__main__':
    app.run(debug=True)
