#importing libraries
##import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import pandas as pd



# Set the working directory to a specific path
#os.chdir("C:/Users/OMEN 16/Workspace Dropbox/Edwin Kochulem/Projects/Flask_RF_model")

loaded_model = pickle.load(open("RF_Land_value_model.pkl","rb"))

preprocessing_pipeline=pickle.load(open("std_pca_scaler.pkl","rb"))

##with open("model.pkl", 'rb') as file:
##    loaded_model = pickle.load(file)

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
    #return "Hello World"


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        # Get the data from the POST request.
        field1 = request.form.get('x1')  # Access the form value
        field1 = int(field1)  # Convert to integer if needed
        df_f1 = pd.DataFrame([[field1]], columns=['x1'])

        field2 = request.form.get('x2')  # Access the form value
        field2 = int(field2)  # Convert to integer if needed
        df_f2 = pd.DataFrame([[field2]], columns=['x2'])

        field3 = request.form.get('x3')  # Access the form value
        field3 = int(field3)  # Convert to integer if needed
        df_f3 = pd.DataFrame([[field3]], columns=['x3'])

        field4 = request.form.get('x4')  # Access the form value
        field4 = int(field4)  # Convert to integer if needed
        df_f4 = pd.DataFrame([[field4]], columns=['x4'])


        field5 = request.form.get('x5')  # Access the form value
        field5 = int(field5)  # Convert to integer if needed
        df_f5 = pd.DataFrame([[field5]], columns=['x5'])


        field6 = request.form.get('x6')  # Access the form value
        field6 = int(field6)  # Convert to integer if needed
        df_f6 = pd.DataFrame([[field6]], columns=['x6'])


        field7 = request.form.get('x7')  # Access the form value
        field7 = int(field7)  # Convert to integer if needed
        df_f7 = pd.DataFrame([[field7]], columns=['x7'])


        field8 = request.form.get('x8')  # Access the form value
        field8 = int(field8)  # Convert to integer if needed
        df_f8 = pd.DataFrame([[field8]], columns=['x8'])


        field9 = request.form.get('x9')  # Access the form value
        field9 = int(field9)  # Convert to integer if needed
        df_f9 = pd.DataFrame([[field9]], columns=['x9'])

        field10 = request.form.get('x10')  # Access the form value
        field10 = int(field10)  # Convert to integer if needed
        df_f10 = pd.DataFrame([[field10]], columns=['x10'])

        field11 = request.form.get('x11')  # Access the form value
        field11 = int(field11)  # Convert to integer if needed
        df_f11 = pd.DataFrame([[field11]], columns=['x11'])


        field12 = request.form.get('x12')  # Access the form value
        field12 = int(field12)  # Convert to integer if needed
        df_f12 = pd.DataFrame([[field12]], columns=['x12'])

        field13 = request.form.get('x13')  # Access the form value
        field13 = int(field13)  # Convert to integer if needed
        df_f13 = pd.DataFrame([[field13]], columns=['x13'])

        Merged_df = pd.concat([df_f1,df_f2,df_f3,df_f4,df_f5,df_f6,df_f7,df_f8,df_f9,df_f10,df_f11,df_f12,df_f13], axis=1)

        df_scaled = preprocessing_pipeline.transform(Merged_df.values)

        # Make prediction using model loaded from disk as per the data.
        prediction1 = loaded_model.predict(df_scaled)

        #Convert Log y to y only using log transform of power e `~`2.718281828
        inve_pred=10**prediction1
        # Take the first value of prediction
        prediction = inve_pred[0]

        return render_template("result.html",prediction=prediction)


if __name__ == "__main__":
    app.run()