# Short-term ADA-Cardano Price Trend Prediction

***
### File Descriptions:
The project contains the following files:

data

* |- Cardano Historical Data.csv # historical daily price data of ADA


Cardano Price Trend Prediction.ipynb: Notebook for exploring data and modeling

README.md

***

### Libraries
To be able to run this notebook, you need to install these libraries:
- Pandas
- Numpy
- Seaborn
- Matplotlib
- sklearn
- xgboost

***
### Project Motivation
For this project I was interested in predicting short-term (1 day) price trend on ADA-Carado Cryptocurrency

The project involved:
 - Loading and cleaning de data from investing.com
 - Conducting Exploratory Data Analysis to understand the data and relevant Features
 - Feature Engineering to build usefull new features to the model.
 - Modelling using machine learning algorithms such as Logistic Regression, Random Forest and XGBoost.

***

### Result Summary
Based on the results and metrics the performance of the model wasn´t as good as expected. However, we obtained better results than an aleatory choice. It is clear that prediction prices and trends in crypto and stock market will continue to be a hot topic as there is a lot to improve and explore.

A Xgboost t Classifier was chosen to be the best model. The final model achieved and Accuracy score of 0.59 and a F1_score of 0.66. 

### Medium Blog Post 
You can read my Medium Blog Post in [here](https://medium.com/@juanchoju/short-term-ada-cardano-price-trend-prediction-da27fb8950e7).


***
###  Acknowledgments 
* The data was downloaded from Investment.com [website](https://www.investing.com/indices/investing.com-btc-usd).
* The cleansing process was inspired based on open code from Kaggle to transform de Volume column: [link](https://www.kaggle.com/jkreyner/script-to-clean-data-set)
* I'd like to acknowledge Udacity for this course and the motivation to do this project.


```python

```
