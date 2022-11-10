# Python program for project - Financial Data Analytics with Python
# Last build date: 2022-8-15
# Objective: Build the streamlit webapp based on Project_final.ipynb

# Import required libraries
# Prepare web scraping data from wikipedia
import bs4 as bs
import requests
#from urllib import request
# Prepare download stock data
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
# Prepare optimization
# Using efficient frontier and assigning weights (Optimization)
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
#from pypfopt.cla import CLA
#from pypfopt import plotting
#from matplotlib.ticker import FuncFormatter
from pypfopt import objective_functions
# For download historial data usiong in optimization
from datetime import timedelta
# Plot charts
import matplotlib.pyplot as plt
# Streamlit library for Web UI
import streamlit as st
# Quantstats library for html report
import quantstats as qs

# Get symbols from Dow Jones wiki page
def get_DJI():
    if source == 'File':
        tickers = ['MMM','AXP','AMGN','AAPL','BA','CAT','CVX','CSCO','KO','DIS','DOW','GS','HD','HON','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PG','CRM','TRV','UNH','VZ','V','WBA','WMT']
    else:
        #Web Scraping for the DJI Tickers
        resp = requests.get('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class' : 'wikitable sortable'}) # table class name is wikitable sortable
        tickers = []
        for row in table.findAll('tr')[1:]:   #Table row
            ticker = row.findAll('td')[1].text.replace('\n','')   #Table data
            ticker = ticker.split(':\xa0')
            if len(ticker) > 1:
                tickers.append(ticker[1])
            else:
                tickers.append(ticker[0])
    return tickers
	
# Or get symbols from S&P100, if your favourite stocks not in DJI, you can use this ticker list
def get_sp100():
    if source == 'File':
        tickers = ['AAPL','ABBV','ABT','ACN','ADBE','AIG','AMGN','AMT','AMZN','AVGO','AXP','BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT','CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC','KO','LIN','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','SCHW','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']
    else:
        # URL request, URL opener, read content
        resp = requests.get('https://en.wikipedia.org/wiki/S%26P_100')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class' : 'wikitable sortable'}) # table class name is wikitable sortable
        tickers = []
        for row in table.findAll('tr')[1:]:   #Table row
            ticker = row.findAll('td')[0].text.replace('\n','')   #Table data
            ticker = ticker.split(':\xa0')
            if len(ticker) > 1:
                tickers.append(ticker[1])
            else:
                tickers.append(ticker[0])
        tickers = list(map(lambda x: x.replace('.', '-'), tickers)) # Yahoo finance compatibility
    return tickers

# Download stock data from web
def get_monthly_data(tickers, start, end):
    # Get data from Yahoo finance, specify the start and end date in {start} and {end} variables
    ''' Downloading Historical Stock Data for all index Stocks '''
    #start = dt.datetime(2020, 6, 1)
    #end = dt.datetime(2022, 6, 1)
    stock_data = pd.DataFrame()
    if source == 'File':
        if stock_index == 'DJI':
            stock_data = pd.read_csv('dji_data_monthly.csv')
        elif stock_index == 'S&P100':
            stock_data = pd.read_csv('sp100_data_monthly.csv')
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
        stock_data = stock_data.set_index('Date')
        stock_data = stock_data[start:end]
    else:
        for stock in tickers:
            #print(f'Downloading stock data for {stock}', end='')
            tickerData = yf.Ticker(stock)
            tickerCl = tickerData.history(interval='1mo', start=start, end=end)['Close']
            stock_data[stock] = tickerCl
            #print(f'......OK!')
        # Data preparation    
        stock_data = stock_data.dropna(how='all')
        stock_data['DOW'].fillna(value=stock_data['DOW'].mean(), inplace=True) # Fill NA values since DOW start from 2019-04
    return stock_data

# From stock data get stock return dataframe
def get_return(stock_data):
    # Convert the stock price to return using pct_change() function
    ''' Calculating the monthly returns of all index Stocks '''
    stock_returns = stock_data.pct_change().dropna()
    return stock_returns

# Prepare historial data for use in optimization algorithm
def get_daily_data(tickers, start):    
    start_hist = start - timedelta(days=365) # 1 yr before the analysis date

    stock_daily_data = pd.DataFrame()
    if source == 'File':
        if stock_index == 'DJI':
            stock_daily_data = pd.read_csv('dji_data_daily.csv')
        elif stock_index == 'S&P100':
            stock_daily_data = pd.read_csv('sp100_data_daily.csv')
        stock_daily_data['Date'] = pd.to_datetime(stock_daily_data['Date'], format='%Y-%m-%d')
        stock_daily_data = stock_daily_data.set_index('Date')
        stock_daily_data = stock_daily_data[start_hist:]
    else:
        for stock in tickers:
            #print(f'Downloading stock data for {stock}', end='')
            tickerData = yf.Ticker(stock)
            tickerCl = tickerData.history(interval='1d', start=start_hist, end=end)['Close']
            stock_daily_data[stock] = tickerCl
            #print(f'......OK!')
        # Data preparation    
        stock_daily_data['DOW'].fillna(value=stock_daily_data['DOW'].mean(), inplace=True)

    return stock_daily_data
    
# For portfolio performance calculation
def volatility(stock_data):
    ''' Calculate the Annualized Volatility of a Trading Strategy '''
    volatility = stock_data['Return'].std() * np.sqrt(12)
    
    return volatility
    
def CAGR(stock_data):
    ''' Calculate the Compound Annual Growth Rate of a Trading Strategy '''
    stock_data['Cumulative_Return'] = (1 + stock_data['Return']).cumprod()
    years = len(stock_data) / 12
    carg = stock_data['Cumulative_Return'].tolist()[-1] ** (1/years) - 1
    
    return carg
    
def sharpe_ratio(stock_data, risk_free):
    ''' Calculate the Sharpe Ratio of a Portfolio '''
    sharpe = (CAGR(stock_data) - risk_free) / volatility(stock_data)
    
    return sharpe
    

# Add new function, determine weights parameters in portfolio (optimization using max sharpe ratio) 
def pf_weights(pf):
    pf_hist = stock_daily_data[pf].loc[:start] # Get stocks price 1 year before the portfolio build date for optimization
    
    #Annualized Return
    mu = expected_returns.mean_historical_return(pf_hist)
    #Sample Variance of Portfolio
    Sigma = risk_models.sample_cov(pf_hist)
    
    #Max Sharpe Ratio - Tangent to the EF
    ef = EfficientFrontier(mu, Sigma, weight_bounds=(min_weight, max_weight))
    #ef.add_objective(objective_functions.L2_reg, gamma=2)
    # Check if the return can apply optimization, otherwise equally distribute the stocks (or try alternative strategy?)
    try:
        if opt_method == 'Max Sharpe':
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif opt_method == 'Min Risk':
            ef.min_volatility()
        elif opt_method == 'Custom Risk' and your_risk.replace('.','',1).isdigit():
            ef.efficient_risk(your_risk)
        elif opt_method == 'Custom Return' and your_return.replace('.','',1).isdigit():
            ef.efficient_return(your_return)
        weights = ef.clean_weights()
    except Exception:
        wt = 1/len(pf)
        weights = dict.fromkeys(pf, wt)
        ef.set_weights(weights)
        
    # print(weights) #DEBUG
    # Convert from dict to dataframe
    weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
    weights_df.columns = ['weights']
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    weights_df['sharpe'] = sharpe_ratio
    weights_df['yr_return'] = expected_annual_return
    weights_df['yr_risk'] = annual_volatility
    return weights_df

# Strategy 1 - Remove n stocks if return is worse than other stock in portfolio
def portfolio(dataframe, n_stocks, n_remove):
    '''
    dataframe: Dataframe with the stocks returns
    n_stocks: Number of stocks to be selected in the portfolio
    n_remove: Number of bad stocks to be remove in the portfolio
    '''
    portfolio = []
    weekly_return = [0]
    # For Displaying
    date_df = []
    portfolio_df = []
    amount = capital
    capital_df = []
    for i in range(1, len(dataframe)):
        if len(portfolio) > 0:
            weekly_return.append(dataframe[portfolio].iloc[i, :].mean())
            amount = amount * (1 + dataframe[portfolio].iloc[i, :].mean())
            negative_stocks = dataframe[portfolio].iloc[i, :].sort_values(ascending=True)[: n_remove].index.tolist()
            portfolio = [stock for stock in portfolio if stock not in negative_stocks]
        to_fill = n_stocks - len(portfolio)
        new_stocks = dataframe.iloc[i,:].sort_values(ascending=False)[:n_stocks].index.tolist()
        new_stocks = [stock for stock in new_stocks if stock not in portfolio][:to_fill] #We make sure to not repeat stocks in the portfolio
        portfolio = portfolio + new_stocks
        # For display
        date_df.append(dataframe.index[i])
        portfolio_df.append(', '.join(portfolio))
        capital_df.append(round(amount,2))
        #print(f'({dji_returns.index[i].strftime("%Y-%m-%d")}) The weekly portfolio selected is {portfolio}')
    #date_show = pd.to_datetime(date_df, format='%Y-%m-%d')
    my_portfolio = pd.DataFrame(portfolio_df, index=date_df, columns=['Portfolio'])
    my_portfolio['Capital'] = capital_df
    my_portfolio['Return'] = np.array(weekly_return)
    portfolio_ret = pd.DataFrame(np.array(weekly_return), columns = ['Return'])
    return portfolio_ret, my_portfolio

# Strategy 2 - Random select n stocks into the portfolio
def portfolio_random(dataframe, n_stocks):
    '''
    dataframe: Dataframe with the stocks returns
    n_stocks: Number of stocks to be selected in the portfolio
    '''
    portfolio = []
    weekly_return = [0]
    # For displaying
    date_df = []
    portfolio_df = []
    amount = capital
    capital_df = []
    # Build portfolio
    for i in range(1, len(dataframe)):
        if len(portfolio) > 0:
            weekly_return.append(portfolio.iloc[i, :].mean())
            amount = amount * (1 + portfolio.iloc[i, :].mean())
        portfolio = dataframe.sample(n=n_stocks, axis='columns')
        date_df.append(dataframe.index[i])
        portfolio_df.append(', '.join(list(portfolio.columns)))
        capital_df.append(round(amount,2))
        #print(f'({dataframe.index[i].strftime("%Y-%m-%d")}) The weekly portfolio selected is {list(portfolio.columns)}')
    my_portfolio = pd.DataFrame(portfolio_df, index=date_df, columns=['Portfolio'])
    my_portfolio['Capital'] = capital_df
    my_portfolio['Return'] = np.array(weekly_return)
    portfolio_ret = pd.DataFrame(np.array(weekly_return), columns = ['Return'])
    return portfolio_ret, my_portfolio

# Strategy 1A - Based on strategy 1 and optimized using Efficient frontier
def portfolio_optimized(dataframe, n_stocks, n_remove):
    '''
    dataframe: Dataframe with the stocks returns
    n_stocks: Number of stocks to be selected in the portfolio
    n_remove: Number of bad stocks to be remove in the portfolio
    '''
    portfolio = []
    weekly_return = [0]
    total_return = [0]
    weekly_sharpe = []
    yr_return = []
    yr_risk = []
    # For displaying
    date_df = []
    portfolio_df = []
    amount = capital
    capital_df = []
    weights_df = []

    for i in range(1, len(dataframe)):
        if len(portfolio) > 0:
            weekly_return.append(np.dot(dataframe[portfolio].iloc[i, :], weights))
            amount = amount * (1 + np.dot(dataframe[portfolio].iloc[i, :], weights)) 
            negative_stocks = dataframe[portfolio].iloc[i, :].sort_values(ascending=True)[: n_remove].index.tolist()
            portfolio = [stock for stock in portfolio if stock not in negative_stocks]
        to_fill = n_stocks - len(portfolio)
        new_stocks = dataframe.iloc[i,:].sort_values(ascending=False)[:n_stocks].index.tolist()
        new_stocks = [stock for stock in new_stocks if stock not in portfolio][:to_fill] #We make sure to not repeat stocks in the portfolio
        portfolio = portfolio + new_stocks
        pf_wt = pf_weights(portfolio)
        weights = pf_wt['weights']
        sharpe = pf_wt['sharpe'][0]
        weekly_sharpe.append(sharpe)
        yr_return.append(pf_wt['yr_return'][0])
        yr_risk.append(pf_wt['yr_risk'][0])
        # For display
        date_df.append(dataframe.index[i])
        portfolio_df.append(', '.join(portfolio))
        capital_df.append(round(amount,2))
        weights_df.append(weights.values.tolist())
        #print(f'({dataframe.index[i].strftime("%Y-%m-%d")}) The weekly portfolio selected is {portfolio}, weights is {weights.values.tolist()}, sharpe is {round(sharpe,2)}')
  
    my_portfolio = pd.DataFrame(portfolio_df, index=date_df, columns=['Portfolio'])
    my_portfolio['Weights'] = weights_df
    my_portfolio['Capital'] = capital_df
    my_portfolio['Return'] = np.array(weekly_return)
    portfolio_ret = pd.DataFrame(np.array(weekly_return), columns = ['Return'])
    portfolio_ret['sharpe'] = pd.DataFrame(np.array(weekly_sharpe))
    portfolio_ret['yr_return'] = pd.DataFrame(np.array(yr_return))
    portfolio_ret['yr_risk'] = pd.DataFrame(np.array(yr_risk))
     
    return portfolio_ret, my_portfolio

# Strategy 2A - Based on strategy 2 and optimized using Efficient frontier

def portfolio_random_optimized(dataframe, n_stocks):
    '''
    dataframe: Dataframe with the stocks returns
    n_stocks: Number of stocks to be selected in the portfolio
    '''
    portfolio = []
    weekly_return = [0]
    total_return = [0]
    weekly_sharpe = []
    yr_return = []
    yr_risk = []
    # For displaying
    date_df = []
    portfolio_df = []
    amount = capital
    capital_df = []
    weights_df = []
    #for i in range(1, 10):
    for i in range(1, len(dataframe)):
        if len(portfolio) > 0:
            weekly_return.append(np.dot(portfolio.iloc[i, :], weights))
            amount = amount * (1 + np.dot(portfolio.iloc[i, :], weights))
        portfolio = dataframe.sample(n=n_stocks, axis='columns')        
        pf_wt = pf_weights(portfolio.columns) #Apply weighting optimization
        weights = pf_wt['weights']
        sharpe = pf_wt['sharpe'][0]
        weekly_sharpe.append(sharpe)
        yr_return.append(pf_wt['yr_return'][0])
        yr_risk.append(pf_wt['yr_risk'][0])
        #print(f'({dji_returns.index[i].strftime("%Y-%m-%d")}) The monthly portfolio selected is {list(portfolio.columns)}, weights is {weights.values.tolist()}, sharpe is {round(sharpe,2)}')
        # For display
        date_df.append(dataframe.index[i])
        portfolio_df.append(', '.join(portfolio.columns))
        weights_df.append(weights.values.tolist())
        capital_df.append(round(amount,2))
        
    my_portfolio = pd.DataFrame(portfolio_df, index=date_df, columns=['Portfolio'])
    my_portfolio['Weights'] = weights_df
    my_portfolio['Capital'] = capital_df
    my_portfolio['Return'] = np.array(weekly_return)

    portfolio_ret = pd.DataFrame(np.array(weekly_return), columns = ['Return'])
    portfolio_ret['sharpe'] = pd.DataFrame(np.array(weekly_sharpe))
    portfolio_ret['yr_return'] = pd.DataFrame(np.array(yr_return))
    portfolio_ret['yr_risk'] = pd.DataFrame(np.array(yr_risk))
    
    return portfolio_ret, my_portfolio
    
def get_market_index():
    # Try to compare above 4 strategies to market index itself
    # Get market index data (same date range as our portfolio) 
    if stock_index == 'DJI':
        if source == 'File':
            mkt_index = pd.read_csv('dji_index_monthly.csv')
            mkt_index['Date'] = pd.to_datetime(mkt_index['Date'], format='%Y-%m-%d')
            mkt_index = mkt_index.set_index('Date')
            mkt_index = mkt_index[start:end]
        else:
            mkt_ticker = yf.Ticker('^DJI')
            mkt_index = mkt_ticker.history(interval='1mo', start=start, end=end)['Close']
    elif stock_index == 'S&P100':
        if source == 'File':
            mkt_index = pd.read_csv('sp100_index_monthly.csv')
            mkt_index['Date'] = pd.to_datetime(mkt_index['Date'], format='%Y-%m-%d')
            mkt_index = mkt_index.set_index('Date')
            mkt_index = mkt_index[start:end]
        else:
            mkt_ticker = yf.Ticker('^OEX')
            mkt_index = mkt_ticker.history(interval='1mo', start=start, end=end)['Close']
    
    # Get return
    mkt_index['Return'] = mkt_index.pct_change()
    mkt_index.dropna(inplace = True)
    return mkt_index

# Calculation the performance
def get_rate(portfolio, risk_free_rate):

    out = 'The Compound Annual Growth Rate is {}%'
    out += '\nThe Portfolio Volality is {}%'
    out += '\nThe Sharpe Ratio is {}'
    return out.format(round(CAGR(portfolio)*100,3), round(volatility(portfolio)*100,3), round(sharpe_ratio(portfolio, risk_free_rate),3))

def generate_quantstats(pf):
    if stock_index == 'DJI':
        qs.reports.html(pf, title='Strategy vs DJI', benchmark='^DJI', output='', download_filename='portfolio_streamlit.html')
    elif stock_index == 'S&P100':
        qs.reports.html(pf, title='Strategy vs S&P 100', benchmark='^OEX', output='', download_filename='portfolio_streamlit.html')

###########################################################
# Main program (For Streamlit)
st.set_page_config(page_title='Portfolio selector', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
#st.markdown(f"""<style>.appview-container .main .block-container{{ max-width: 60%; }}</style>""",unsafe_allow_html=True,)
st.header('Portfolio selection WebApp - Group project')
st.subheader('This web app will generate portfolios based on 4 strategies')

# These are paramters to cons truct a new portfolio with optimization
# 1. Min and max weight of each stock
# 2. Target return
# 3. Target risk
# 4. Min risk
# 5. Max return (no need)

# Below is default one, feel free to change to your values
# Default values (Implement the value in streamlit)
#capital = 10000
#risk_free_rate = 0.02
#min_weight = 0.05
#max_weight = 1
#your_risk = 0
#your_return = 0
#strategy = 'max_sharpe' # (vaild options: max_sharpe / min_risk / custom_risk? / custom_return?)
# User input components
source = st.radio("Data source: ", ('File', 'Web (Not tested, very slow!)'))
stock_index = st.radio("Select your stock index: ", ('DJI', 'S&P100'))
start = st.date_input("Start Date: ", dt.date(2019, 5, 1))
end = st.date_input("End Date: ", dt.date.today())
risk_free_rate = st.number_input('Risk free rate:', min_value=0.00, max_value=1.00, value=0.02, step=0.01)
capital = st.number_input('Initial capital (USD):', min_value=0, value=10000)
n_stocks = st.number_input('Number of stocks in your portfolio:', min_value=1, value=8, step=1)
strategy = st.radio("Select your portfolio strategy: ", ('Remove n worst stock', 'Random stock pick', 'Remove n worst stock and do optimization', 'Random stock pick and do optimization'))

# Additional options if choosing different strategy
if strategy == 'Remove n worst stock':
    n_remove = st.number_input('Number of stocks (n) removed in each month:', min_value=1, value=3, step=1)
elif strategy == 'Remove n worst stock and do optimization':
    opt_method = st.radio("Select your optimization method: ", ('Max Sharpe', 'Min Risk', 'Custom Risk', 'Custom Return'))
    n_remove = st.number_input('Number of stocks removed in each period:', min_value=1, value=3, step=1)
    min_weight = st.number_input('Min weight:', min_value=0.00, max_value=1.00, value=0.05, step=0.01)
    max_weight = st.number_input('Max weight:', min_value=0.00, max_value=1.00, value=1.00, step=0.01)
    if opt_method == 'Custom Risk':
        your_risk = st.number_input('Custom risk:', min_value=0.00, max_value=1.00, value=0.1, step=0.01)
    elif opt_method == 'Custom Return':
        your_return = st.number_input('Custom return:', min_value=0.00, max_value=1.00, value=0.8, step=0.01)
elif strategy == 'Random stock pick and do optimization':
    opt_method = st.radio("Select your optimization method: ", ('Max Sharpe', 'Min Risk', 'Custom Risk', 'Custom Return'))
    min_weight = st.number_input('Min weight:', min_value=0.00, max_value=1.00, value=0.05, step=0.01)
    max_weight = st.number_input('Max weight:', min_value=0.00, max_value=1.00, value=1.00, step=0.01)
    if opt_method == 'Custom Risk':
        your_risk = st.number_input('Custom risk:', min_value=0.00, max_value=1.00, value=0.1, step=0.01)
    elif opt_method == 'Custom Return':
        your_return = st.number_input('Custom return:', min_value=0.00, max_value=1.00, value=0.8, step=0.01)

report = st.radio("View report using Quantstats? (Require internet connection)", ('Yes', 'No'))
# Button for build and display portfolio performance
if st.button('Backtesting your portfolio'):
    #st.session_state.more_stuff = True
    # Get stock price data
    if stock_index == 'DJI':
        tickers = get_DJI()
        stock_data = get_monthly_data(tickers, start, end)
        stock_daily_data = get_daily_data(tickers, start)  
        index_data = get_market_index()        
    elif stock_index == 'S&P100':
        tickers = get_sp100()
        stock_data = get_monthly_data(tickers, start, end)
        stock_daily_data = get_daily_data(tickers, start)  
        index_data = get_market_index()
  
    stock_returns = get_return(stock_data)
    # Run def based on selected strategy 
    if strategy == 'Random stock pick':
        pf_return, pf = portfolio_random(stock_returns, n_stocks)
    elif strategy == 'Random stock pick and do optimization':
        pf_return, pf = portfolio_random_optimized(stock_returns, n_stocks)
    elif strategy == 'Remove n worst stock':
        pf_return, pf = portfolio(stock_returns, n_stocks, n_remove)
    elif strategy == 'Remove n worst stock and do optimization':
        pf_return, pf = portfolio_optimized(stock_returns, n_stocks, n_remove)
    
    # Display the portfolio details
    ''' Portflio Details'''
    pf_show = pf.copy() # Prevent modify the original data
    pf_show.index = pf_show.index.strftime('%Y-%m-%d')
    st.write(pf_show[['Portfolio','Capital']])
    #st.write(pf_return['Return'])
    # Display the performance
    '''Portfolio risk and return data'''
    st.text(get_rate(pf_return, risk_free_rate))
    
    '''Market data'''
    st.text(get_rate(index_data, risk_free_rate))
    
    # Plotting graph for comparsion
    # Prepare
    if source == 'File':
        mkt_start = 1
    else:
        mkt_start = 2
    # Plot the index and our portfolio performance, check if better than market index or not
    ''' Visualization '''

    fig, ax = plt.subplots(figsize = (20, 10))
    plt.plot((1 + pf_return['Return'][:-2]).cumprod(), color = 'g')
    plt.plot((1 + index_data['Return'][mkt_start:-2].reset_index(drop = True)).cumprod(), color = 'r')
    ax.legend(['Strategy', 'Market return'], fontsize = 15)
    plt.title('Your strategy compare with market index', fontsize = 20)
    plt.ylabel('Cumulative Return', fontsize = 20)
    plt.xlabel('Month', fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    plt.subplots_adjust(top = 0.92, bottom = 0.1, left = 0.08, right = 0.94)
    st.pyplot(fig)
    
    # Display build portfolio in the future (month of the end date) 
    '''If you accept the back test result above, you can invest following portfolio in the coming month:'''
    #new_pf = pd.DataFrame(pf['Portfolio'][-1], columns=['Stocks'])
    #new_pf['Weights'] = pf['Weights'][-1]
    st.subheader(pf['Portfolio'][-2])
    if strategy == 'Random stock pick and do optimization' or strategy == 'Remove n worst stock and do optimization':
        '''With this weighting:'''
        st.subheader(', '.join(map(str,pf['Weights'][-2])))
        
    #pf['Return']
    if report == 'Yes':
        st.text('-----------------------------------------------------------')
        st.subheader('Portfolio analytics using Quantstats')
        pf.index = pf.index + pd.DateOffset(months=1)
        generate_quantstats(pf['Return'][:-2])
        report_file = open("portfolio_streamlit.html", 'r', encoding='utf-8')
        html_code = report_file.read()
        #st.write(html_code)
        st.components.v1.html(html_code, width=1050, height=4500, scrolling=False)
        #generate_quantstat(pf['Return'])
        #st.components.v1.html(qs.reports.full(pf['Return']))
        #st.write(qs.reports.full(pf['Return']))
        #st.components.v1.iframe('http://localhost:8501/portfolio_vs_market.html')
    
    
# End Main Program    
#############################################