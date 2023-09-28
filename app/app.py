from flask import Flask, request, render_template, jsonify
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file.save(file.filename)
        fig1 = perform_analysis_and_plot1(file.filename)
        fig2 = perform_analysis_and_plot2(file.filename)
        return jsonify(fig1=fig1, fig2=fig2)






def perform_analysis_and_plot1(filepath):
    fig = go.Figure()
    df = pd.read_excel(filepath)
    omega = df['omega']
    
    for i in range(5):  # adding 5 traces to the plot
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1+i, 3, 2+i], name=f'Trace {i+1}'))
    return fig.to_json()

def perform_analysis_and_plot2(filepath):
    fig = go.Figure()
    for i in range(5):  # adding 5 traces to the plot
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1+i, 3, 2+i], name=f'Trace {i+1}'))
    return fig.to_json()

if __name__ == '__main__':
    app.run(debug=True)
