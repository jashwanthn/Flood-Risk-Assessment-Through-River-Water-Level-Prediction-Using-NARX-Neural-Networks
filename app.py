import csv
from flask import Flask, jsonify, render_template,request,make_response
import mysql.connector
from mysql.connector import Error
import sys
import os, random
import pygame
import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
from sklearn.pred import *
from skimage import measure #scikit-learn==0.23.0
#from skimage.measure import structural_similarity as ssim #old
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob


app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def index1():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')



@app.route('/mainpage', methods = ["GET","POST"])
def mainpage():
    if request.method == "GET":
      connection = mysql.connector.connect(host='localhost',database='flood_2024',  user='root',password='') 
      cursor = connection.cursor()
      sq_query="select * from predictdata limit 1"  
      cursor.execute(sq_query)
      data = cursor.fetchall()
      num_rows = len(data) 
      cursor.close()
      connection.close() 
      return render_template('mainpage.html', rows = num_rows)
    
    else:
      connection = mysql.connector.connect(host='localhost',database='flood_2024',  user='root',password='') 
      cursor = connection.cursor()
      sq_query="TRUNCATE TABLE predictdata;" 
      cursor.execute(sq_query) 
      cursor.close()
      connection.close() 
      return make_response(json.dumps("SUCCESS"))


@app.route('/predict', methods =  ['GET','POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        heavy_rainfall = request.form.get('heavy_rainfall', '').strip()
        rainfall_duration = request.form.get('rainfall_duration', '').strip()
        drainage_issue = request.form.get('drainage_issue', '').strip()
        flood_warning = request.form.get('flood_warning', '').strip()
        above_avg_rainfall_season = request.form.get('above_avg_rainfall_season', '').strip()
        highest_rainfall_month_extreme = request.form.get('highest_rainfall_month_extreme', '').strip()

        if not all([heavy_rainfall, rainfall_duration, drainage_issue, flood_warning, above_avg_rainfall_season, highest_rainfall_month_extreme]):
            return render_template("predict.html", error="All fields are required.")
        result = predict_flood(heavy_rainfall, rainfall_duration, drainage_issue, flood_warning,
                               above_avg_rainfall_season, highest_rainfall_month_extreme)

        if result is None:
            return render_template("predict.html", error="Invalid input values.")
        
        return render_template("predict.html",
                               prediction=result[0],
                               probability=result[1],
                               flood_type=result[2],
                               accuracy=result[3])
         
    return render_template("predict.html", prediction=None)


@app.route('/forecast', methods =  ['GET','POST'])
def forecast():
  
  if request.method == "GET":
    connection = mysql.connector.connect(host='localhost',database='flood_2024',user='root',password='') 
    cursor = connection.cursor()
    sq_query="select * from predictdata limit 500"
    # sq_query="select * from predictdata"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    cursor.execute("SHOW COLUMNS FROM predictdata")
    columns = [column[0] for column in cursor.fetchall()]
    cursor.close()
    connection.close()  
    
    return render_template('forecast.html',data = data,columns = columns)


'''Register Code'''
@app.route('/regdata', methods =  ['GET','POST'])
def regdata():
    connection = mysql.connector.connect(host='localhost',database='flood_2024',user='root',password='')
    uname = request.args['uname']
    email = request.args['email']
    phn = request.args['phone']
    pssword = request.args['pswd']
    addr = request.args['addr']
    dob = request.args['dob']
    print(dob)
        
    cursor = connection.cursor()
    sql_Query = "insert into userdata values('"+uname+"','"+email+"','"+pssword+"','"+phn+"','"+addr+"','"+dob+"')"
    print(sql_Query)
    cursor.execute(sql_Query)
    connection.commit() 
    connection.close()
    cursor.close()
    msg="User Account Created Successfully"    
    resp = make_response(json.dumps(msg))
    return resp

 


"""LOGIN CODE """

@app.route('/logdata', methods =  ['GET','POST'])
def logdata():
    connection=mysql.connector.connect(host='localhost',database='flood_2024',user='root',password='')
    lgemail=request.args['email']
    lgpssword=request.args['password']
    print(lgemail, flush=True)
    print(lgpssword, flush=True)
    cursor = connection.cursor()
    sq_query="select count(*) from userdata where Email='"+lgemail+"' and Pswd='"+lgpssword+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)
    
    connection.commit() 
    connection.close()
    cursor.close()
    
    if rcount>0:
        msg="Success"
        resp = make_response(json.dumps(msg))
        return resp
    else:
        msg="Failure"
        resp = make_response(json.dumps(msg))
        return resp
        

@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST': 
        connection = mysql.connector.connect(host='localhost',database='flood_2024',user='root',password='')
        cursor = connection.cursor()
    
        prod_mas = request.files['first_image']
        filename = secure_filename(prod_mas.filename)
        prod_mas.save(os.path.join(".\\static\\uploads\\", filename))


        fn = os.path.join(".\\static\\uploads\\", filename)


        fields = [] 
        rows = []
        
        with open(fn, 'r') as csvfile:
            
            csvreader = csv.reader(csvfile)   
  
            for row in csvreader:
                rows.append(row)
                print(row)

        try:     
         for row in rows[1:]: 
           if row[0][0]!="":                
            query=""
            query="INSERT INTO predictdata(`SUBDIVISION`, `YEAR`, `JAN`, `FEB`, `MAR`, `APR`, `MAY`, `JUN`, `JUL`,`AUG`,`SEP`,`OCT`,`NOV`,`DEC`,`ANNUAL RAINFALL`,`FLOODS`) VALUES ("
            for i, col in enumerate(row): 
                if i == 4 and col == "":
                    col = "0"  
                query = query + '"' + col + '",'
            query = query[:-1]
            query = query + ");"
            print("query :" + str(query), flush=True)
            cursor.execute(query)
            connection.commit()
        except Exception as e:
         print("An exception occurred")
         print(e)

        csvfile.close()
        
        print("Filename :"+str(prod_mas), flush=True)       
        
        
        connection.close()
        cursor.close()
 
        resp = make_response(json.dumps("success"))
        return resp

def nn_predict(params, X, input_size, hidden_size):
    # Unpack parameters:
    # W1: shape (hidden_size, input_size)
    # b1: shape (hidden_size,)
    # W2: shape (hidden_size,)
    # b2: scalar
    idx1 = hidden_size * input_size
    W1 = params[:idx1].reshape(hidden_size, input_size)
    b1 = params[idx1:idx1+hidden_size]
    idx2 = idx1 + hidden_size
    W2 = params[idx2:idx2+hidden_size]
    b2 = params[idx2+hidden_size]
    
    # Hidden layer computation with tanh activation
    hidden_input = np.dot(X, W1.T) + b1
    hidden_output = np.tanh(hidden_input)
    
    # Linear output layer
    y_pred = np.dot(hidden_output, W2) + b2
    return y_pred

# Error function used by leastsq
def error_function(params, X, y_true, input_size, hidden_size):
    y_pred = nn_predict(params, X, input_size, hidden_size)
    return y_pred - y_true

def train_narx_model(N=200, seed=0):
    """
    Trains a simple NARX model using synthetic data.
    
    Parameters:
      N (int): Total number of time steps for the synthetic data.
      seed (int): Random seed for reproducibility.
      
    Returns:
      trained_params: The trained neural network parameters.
      cov_x: Covariance of the parameters (from leastsq).
      infodict: Information dictionary from leastsq.
      mesg: Message from leastsq.
      ier: Integer flag from leastsq.
      X: The generated feature matrix.
      y_true: The generated target vector.
    """
    np.random.seed(seed)
    
    # Generate synthetic rainfall data (exogenous input)
    rainfall = np.abs(np.random.randn(N))  # ensure nonnegative rainfall values

    # Generate synthetic flood levels using a simple dynamic model
    true_flood = np.zeros(N)
    for t in range(2, N):
        # Simple dynamic model: 
        # flood(t) = 0.5*flood(t-1) - 0.2*flood(t-2) + 0.3*rainfall(t-1) + noise
        true_flood[t] = 0.5 * true_flood[t-1] - 0.2 * true_flood[t-2] \
                        + 0.3 * rainfall[t-1] + 0.1 * np.random.randn()

    # Prepare dataset for NARX model using a lag of 2 time steps
    lag = 2
    X_list = []
    y_list = []
    for t in range(lag, N):
        # Input vector: [flood(t-1), flood(t-2), rainfall(t-1), rainfall(t-2)]
        x_t = [true_flood[t-1], true_flood[t-2], rainfall[t-1], rainfall[t-2] if t-2 >= 0 else 0]
        X_list.append(x_t)
        y_list.append(true_flood[t])
    
    X = np.array(X_list)         # shape: (N - lag, 4)
    y_true = np.array(y_list)      # shape: (N - lag,)

    # Neural network architecture
    input_size = X.shape[1]      # should be 4 in this case
    hidden_size = 5              # number of neurons in the hidden layer

    # Total parameters: W1 (hidden_size x input_size) + b1 (hidden_size) + W2 (hidden_size) + b2 (1)
    n_params = hidden_size * input_size + hidden_size + hidden_size + 1
    initial_params = np.random.randn(n_params) * 0.1

    # Train the network using Levenberg–Marquardt algorithm
    result = leastsq(error_function, initial_params, args=(X, y_true, input_size, hidden_size), full_output=True)
    trained_params, cov_x, infodict, mesg, ier = result

    return trained_params, cov_x, infodict, mesg, ier, X, y_true



if __name__ == '__main__':
    app.run(host='0.0.0.0')
