# Data Scientist Nanodegree
***
## Disaster Response Pipeline Project
***

## Table of Contents

- [Project Description](#overview)
- [Files Descripton](#description)
- [Components](#components)
- [Running](#run)
- [Results](#results)
***

<a id='overview'></a>
### Project Description:

This project is part of the Data Science Nano Degree by Udacity and the objective is to apply  the data engineering  skills acquired during the course in order to analyze disaster data from *Figure Eight* to build a model for an API that classifies disaster messages. Furthermore, the project include a web app to deploy all the results trhough a Flask application
 The landing page of the web app includes 3 visualizations of a training dataset built using plotly.

***
<a id='description'></a>
### File Descriptions:
The project contains the following files,

app

| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # data cleaning pipeline
|- InsertDatabaseName.db # database to save clean data to

models

|- train_classifier.py # machine learning pipeline

ETL Pipeline Preparation.ipynb: Notebook experiment for the ETL pipelines
ML Pipeline Preparation.ipynb: Notebook experiment for the machine learning pipelines

README.md

***
<a id='components'></a>
### Components
There are three components I completed for this project. **This information was copied directly from the Udacity course instruction** 

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline that:

 - Loads the messages and categories datasets
 - Merges the two datasets
 - Cleans the data
 - Stores it in a SQLite database
 
A jupyter notebook `ETL Pipeline Preparation` was used to do EDA to prepare the process_data.py python script. 
 
#### 2. ML Pipeline
A Python script, `train_classifier.py`, writes a machine learning pipeline that:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file
 
A jupyter notebook `ML Pipeline Preparation` was used to do EDA to prepare the train_classifier.py python script. 

#### 3. Flask Web App
The project includes a deploied Web App which includes a data visualizations using Plotly

***
<a id='run'></a>
### Running:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app.py`

3. Go to http://0.0.0.0:3001/

***
<a id='results'></a>
### Results:
![Webapp Screenshot](https://github.com/jdjaramillou/Data-Scientist-Nanodegree-Program/blob/main/Project2/screenshot/ss1.PNG?raw=true)

![Webapp Screenshot](https://github.com/jdjaramillou/Data-Scientist-Nanodegree-Program/blob/main/Project2/screenshot/ss2.PNG?raw=true)

![Webapp Screenshot](https://github.com/jdjaramillou/Data-Scientist-Nanodegree-Program/blob/main/Project2/screenshot/ss3.PNG?raw=true)



```python

```
