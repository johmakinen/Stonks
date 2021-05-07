import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as pdr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
# from tqdm.notebook import tqdm
import yfinance as yf
import random

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import tempfile

# import tensorflow as tf
# from tensorflow import keras

import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

def get_fundamentals(tickers):
    '''Gets the fundamentals data for given tickers and produces a clean dataframe from it'''
    
    tickers_data = {}
    # fundamentals = ['forwardPE',
    #                 'trailingPE',
    #                 'forwardEps',
    #                 'sector',
    #                 'fullTimeEmployees',
    #                 'country',
    #                 'twoHundredDayAverage',
    #                 'averageDailyVolume10Day',
    #                 'trailingPE',
    #                 'marketCap',
    #                 'priceToSalesTrailing12Months',
    #                 'trailingEps',
    #                 'priceToBook',
    #                 'earningsQuarterlyGrowth',
    #                 'pegRatio']
    filter_date = dt.datetime.today()-dt.timedelta(weeks=2)
    results_dict = {}
    # Loop all tickers and get some interesting fundamentals.
    # tickers = ["GOOGL","AMZN","FB"] #<- for testing before going for the 1 hour update of all sp500 tickers
    # for ticker in tqdm(tickers):
    for ticker in tickers:

        ticker_object = yf.Ticker(ticker)
        # print(ticker)
        # Get the recommendations
        tickers_recs_all = ticker_object.recommendations
        if tickers_recs_all is not None:
            latest_recs = tickers_recs_all.loc[tickers_recs_all.index>=filter_date,"To Grade"]
            if not latest_recs.empty:
                rec = latest_recs.mode()
                if len(rec.index) == 1:
                    results_dict[ticker] = rec.item()
                else:
                    results_dict[ticker] = rec.loc[0]


        #convert info() output from dictionary to dataframe
        # new_info = { key:value for (key,value) in ticker_object.info.items() if key in fundamentals}
        new_info = { key:value for (key,value) in ticker_object.info.items()}

        temp = pd.DataFrame.from_dict(new_info, orient="index")
        temp.reset_index(inplace=True)
        if len(temp.columns) == 2:
            temp.columns = ["Attribute", "Value"]
            # add (ticker, dataframe) to main dictionary
            tickers_data[ticker] = temp

    # Recommendation data into neat dataframe
    results_df = pd.DataFrame.from_dict(results_dict,orient="index").reset_index().rename(columns={"index":'Ticker',0:'recommendation'})

    # Info data into neat dataframe
    combined_data = pd.concat(tickers_data).reset_index().drop(columns="level_1").rename(columns={'level_0': 'Ticker'})
    combined_data = combined_data.pivot(index='Ticker', columns='Attribute', values='Value').reset_index()
    combined_data = combined_data.rename_axis(None, axis=1).infer_objects()
    # combined_data.dropna(inplace=True) # Drop if any fundamental is NA

    combined_data = combined_data.merge(results_df,how="left")
    return combined_data

def get_data(mode="test",update_csv=False):
    '''Fetches stock tickers and fundamentals data from Yahoo or csv'''
    if mode == "test":
        # Tickers for lighter computing
        tickers =['FB','AMZN', 'AAPL', 'NFLX', 'GOOGL', 'MSFT']
        fundamentals = get_fundamentals(tickers)
    elif mode == "all":
        #Get all tickers from csv, if no csv in directory -> scrape them from wikipedia
        SP500_fileName = "SP500_symbols.csv"
        if not os.path.isfile(SP500_fileName):
            tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            tickers = tickers[0]["Symbol"]
            tickers.to_csv(SP500_fileName)
        else:
            tickers = pd.read_csv(SP500_fileName).drop(['Unnamed: 0'],axis=1).to_numpy().flatten()

        # Get all fundamentals from csv, if no csv in directory -> scrape them from yahoo
        fundamentals_fileName = "SP500_fundamentals.csv"
        if (not os.path.isfile(fundamentals_fileName)) or update_csv:
            fundamentals = get_fundamentals(tickers)
            fundamentals.to_csv(fundamentals_fileName)
        else:
            fundamentals = pd.read_csv(fundamentals_fileName).drop(['Unnamed: 0'],axis=1).rename(columns={'majority_recommendation':'recommendation'})
    else:
        print("Select mode")
        return 0

    return tickers,fundamentals[fundamentals["Ticker"] != "UDR"] # Remove UDR from data as a huge outlier


def monitor_stock(stockName,start_date = "2020-01-01"):
    '''Creates an interactive Plotly figure to monitor the share prices and volumes of given stocks'''

    start = dt.datetime.strptime(start_date, '%Y-%m-%d')
    end = dt.datetime.now()
    stock_df = pdr.DataReader(stockName, 'yahoo', start, end)
    # stocks.describe()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.03, 
               row_width=[0.2, 0.7])

    # Old, used when there are multiple stocks in the df
    # fig.add_trace(go.Candlestick(x = stock_df.index, 
    #                                                open = stock_df[('Open',    stockName)], 
    #                                                high = stock_df[('High',    stockName)], 
    #                                                low = stock_df[('Low',    stockName)], 
    #                                                close = stock_df[('Close',    stockName)],showlegend=False,name="Price"))


    fig.add_trace(go.Candlestick(x = stock_df.index, 
                                                   open = stock_df['Open'], 
                                                   high = stock_df['High'], 
                                                   low = stock_df['Low'], 
                                                   close = stock_df['Close'],showlegend=False,name="Price"))

    fig.update_xaxes(row=1, col=1,
        title_text = '',
        rangeslider_visible = False,
        rangeselector = dict(
            buttons = list([
                dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
                dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
                dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
                dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
                dict(step = 'all')])))
    
    fig.add_trace(go.Bar(x = stock_df.index,
                        y=stock_df['Volume'],
                        showlegend=False,name="Volume",
                        marker=dict(color="rgba(0,0,0.8,0.66)")),row=2, col=1)

 
    
    fig.update_layout(
        width=1280,
        height=800,
        title = {
            'text': stockName +' STOCK MONITOR',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            plot_bgcolor =  "rgba(1,1,1,0.05)")
    
    fig.update_yaxes(title_text ='Close Price', tickprefix = '$',row=1,col=1)
    fig.update_yaxes(title_text = 'Volume',row=2,col=1)
    fig.show()

def plot_boxes(data):
   fig = px.box(data.melt(id_vars=["Ticker"]),
   y="value",
   facet_col="variable",
      color="variable",
      boxmode="overlay",
      hover_name="Ticker")

   fig.update_layout(width=1280,
                     height=600,
                     showlegend=False)
   fig.update_yaxes(matches=None,showticklabels=True)
   fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
   fig.show()


def remove_outliers(data,q=0.999):
    '''Removes huge outliers that do not belong in the 99.9 percentile'''

    for column in data.columns:
        if not isinstance(data[column].iloc[0],str):
            q_hi = data[column].quantile(q)
            q_low = data[column].quantile(1-q)

            data = data[(data[column]<q_hi) & (data[column] > q_low)]
    return data

def pca_on_fundamentals(data):
    '''Performs PCA on the numeric values of the fundamentals dataset'''
    features = data.select_dtypes(include=np.number).columns.tolist()
    x = data.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    pd.DataFrame(data = x, columns = features).head()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
    # print("Explained variance ratios: ",pca.explained_variance_ratio_)
    return principalDf

def plot_pca(data):
    '''Plots the PCA onto two dimensions using interactive Plotly scatterplot'''
    principalDf = pca_on_fundamentals(data)

    fig = px.scatter(principalDf,
    x="PC1",
    y="PC2",
    color=data["recommendation"])

    fig.update_layout(
        width=1280,
        height=800,
        title = {
            'text': 'Scatter plot of the principal components',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.show()

def plot_confusion(labels, predictions):
  cm = confusion_matrix(labels, predictions.argmax(1))
  plt.figure()
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix - argmax from predicted classes')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')