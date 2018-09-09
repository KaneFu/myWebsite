import datetime
import json

import jieba
import numpy as np
import pandas as pd
import statsmodels.api as sm
from bs4 import BeautifulSoup
from flask import Flask, render_template,request
from flask_bootstrap import Bootstrap
from scipy import stats
from gensim import corpora
from gensim.models import TfidfModel
from pyecharts import WordCloud
from pymongo import MongoClient


from tools import *

app = Flask(__name__)
app.config['SECRET_KEY'] = "IM your Dad"
bootstrap = Bootstrap(app)
client = MongoClient()
collection = client.research.newTable

factor_types = ['partNum','OrgNum','QuestionNum','Dummy']
factor_type = factor_types[3]

## industry_list
global INDUSTRY_LIST
INDUSTRY_LIST = collection.find({}).distinct("stkIndustry")
global STK_INFO
global STK_INFO2
STK_INFO = pd.read_excel('utils/股票信息词典.xlsx',encoding='gbk',index_col=2)
STK_INFO2 =pd.read_excel('utils/股票信息词典.xlsx',encoding='gbk', index_col=0)
global DAILY_FACTOR_DF
DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)

global MONTHLY_FACTOR_DF
MONTHLY_FACTOR_DF = pd.read_csv('utils/factor_monthly_raw.csv',index_col=0,parse_dates = True)

global MKT_CAP
MKT_CAP = pd.read_csv("utils/monthly_cap.csv",index_col=0,parse_dates = True)

global STK_RET
STK_RET = pd.read_csv("utils/monthly_return.csv",index_col=0,parse_dates = True)

global MOMENTUM
MOMENTUM = pd.read_csv('utils/other_factors/momentum3M.csv',index_col=0, parse_dates=True)

global PB
PB = pd.read_csv('utils/other_factors/monthly_pb.csv',index_col=0, parse_dates=True)

global MKT_CAP_FACTOR
df = np.log(MKT_CAP).apply(truncate,axis=1,result_type='broadcast')
df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')
df = df['20130101':'20180701']
MKT_CAP_FACTOR = df

std_index_ret = pd.read_csv('utils/std_index_ret.csv',index_col=0, parse_dates=True)
######IC衰减窗口


########每个月度做回归，并且画出来月度的因子散点图来看
# avg_RankIC = []
# IC_IR = []
# abs_T_greater2 = []
# factor_return = []
# generate_factor_tool(600,10,factor_type=factor_type)

# for delay_month in range(12):
#     DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)
#     # MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()
#     decay_window = 12
#     decay_factor = 2
#     time_std_window = 12
#     delay_len = 2
#     df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum()
#     log_mkt_cap = np.log(MKT_CAP)
#     industry_list = INDUSTRY_LIST
#     ##衰减
#     df = df.rolling(decay_window).apply(decay, args=(decay_factor,))
#     ## 时序标准化
#     df = df.rolling(time_std_window).apply(lambda x: 0 if np.nanstd(x)==0 else (x[-1]-np.nanmean(x))/np.nanstd(x))
#     ######此时的因子覆盖率
#     cover_rate = df.apply(lambda x: np.sum(x!=0)/len(x), axis=1)
#     ##缩极值
#     df = df.apply(truncate,axis=1,result_type='broadcast')
#     log_mkt_cap = log_mkt_cap.apply(truncate,axis=1,result_type='broadcast')
#     ##横截面标准化
#     df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')
#     log_mkt_cap = log_mkt_cap.apply(lambda x: (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')
    
#     stk_ret = STK_RET.reindex(columns=df.columns)
#     df = df + stk_ret - stk_ret
#     df = df['20130101':'20180701']

#     log_mkt_cap = log_mkt_cap['20130101':'20180701']

#     industry_dummy = pd.DataFrame(np.zeros((len(df.columns),len(industry_list))),index=df.columns,columns=industry_list)
#     ####保证stk_info全面性
#     for stk in industry_dummy.index:
#         industry_dummy.loc[stk, STK_INFO.loc[stk,'industry_sw']] = 1
#     industry_dummy = industry_dummy.drop(columns=industry_list[-1])

#     T_values = {}
#     Adj_Rvalues = {}
#     Num_Cap_coef = {}
#     Factor_Ret = {}
#     IC = {}
#     Rank_IC = {}

#     titles = []
#     dates = df.index
#     for date_idx in range(len(dates)-1-delay_month):
#         date = dates[date_idx]
#         tomorrow = dates[date_idx+1+delay_month]
#         trading_stks = df.columns[~pd.isnull(df.loc[date])]
#         stk_returns_day = stk_ret.loc[tomorrow,trading_stks]*0.01
#         factor_day = df.loc[date,trading_stks].rename("num_factor")
        
#         mkt_cap_day = log_mkt_cap.loc[date,trading_stks].rename("mkt_cap")
#         industry_day = industry_dummy.loc[trading_stks]
#         X_df = pd.concat((factor_day,mkt_cap_day,industry_day),axis=1)
#         X_df = X_df.dropna(axis=0,how='any')
   
#         ###去残差后的IC
#         model = sm.OLS(X_df.iloc[:,0],X_df.iloc[:,1:])
#         results = model.fit()
#         resid_factors = results.resid
#         factor_return_df = pd.concat((resid_factors,stk_returns_day),axis=1)
#         IC[tomorrow] = factor_return_df.corr("pearson").iloc[0,1]
#         Rank_IC[tomorrow] = factor_return_df.corr("spearman").iloc[0,1]



#     Info_df = pd.DataFrame({'IC':IC,'Rank_IC':Rank_IC,})
#     #####RankIC分布图
#     Rank_IC = Info_df['Rank_IC']


#     avg_RankIC.append(np.nanmean(Rank_IC))
#     IC_IR.append( np.nanmean(Rank_IC)/np.nanstd(Rank_IC))


# IC_decay = pd.DataFrame({'avgIC':np.array(avg_RankIC)*100, 'IC_IR':IC_IR}, index=range(12))
# IC_decay.to_csv('utils/sensitive/%s_IC_decay_rsns.csv' % factor_type)


#######调研人次截断敏感性分析
avg_RankIC = []
IC_IR = []
abs_T_greater2 = []
factor_return = []
port_return = []
params = [6,8,10,12,15,20,25,30,35,40]
for param in params:
    generate_factor_tool(600,param,factor_type=factor_type)
    DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)
    STOCK_POOL = DAILY_FACTOR_DF.columns.tolist()
    # MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()
    decay_window = 12
    decay_factor = 2
    time_std_window = 12
    delay_len = 2

    df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum()
    ##衰减
    df = df.rolling(decay_window).apply(decay, args=(decay_factor,))
    ## 时序标准化
    df = df.rolling(time_std_window).apply(lambda x: 0 if np.nanstd(x)==0 else (x[-1]-np.nanmean(x))/np.nanstd(x))
    ######此时的因子覆盖率
    cover_rate = df.apply(lambda x: np.sum(x!=0)/len(x), axis=1)
    ##缩极值
    df = df.apply(truncate,axis=1,result_type='broadcast')

    ##横截面标准化
    df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')   
    stk_ret = STK_RET.reindex(columns=df.columns)
    df = df + stk_ret - stk_ret
    df = df['20130101':'20180701']

    # 先进行回归
    industry_dummy = pd.DataFrame(np.zeros((len(df.columns),len(industry_list))),index=df.columns,columns=industry_list)
    ####保证stk_info全面性
    for stk in industry_dummy.index:
        industry_dummy.loc[stk, STK_INFO.loc[stk,'industry_sw']] = 1
    industry_dummy = industry_dummy.drop(columns=industry_list[-1])
    
    ########每个月度做回归，并且画出来月度的因子散点图来看
    T_values = {}
    Adj_Rvalues = {}
    Num_Cap_coef = {}
    Factor_Ret = {}
    IC = {}
    Rank_IC = {}
    factor_scatter_list = []
    titles = []
    dates = df.index
    for date_idx in range(len(dates)-1):
        date = dates[date_idx]
        tomorrow = dates[date_idx+1]
        trading_stks = df.columns[~pd.isnull(df.loc[date])]
        stk_returns_day = stk_ret.loc[tomorrow,trading_stks]*0.01
        factor_day = df.loc[date,trading_stks].rename("num_factor")
        factor_scatter_list.append(pd.DataFrame({'factor':factor_day, 'return':stk_returns_day}).values.tolist())
        
        mkt_cap_day = MKT_CAP_FACTOR.loc[date,trading_stks].rename("mkt_cap")
        industry_day = industry_dummy.loc[trading_stks]
        X_df = pd.concat((factor_day,mkt_cap_day,industry_day),axis=1)
        model = sm.OLS(stk_returns_day,X_df)
        # model = sm.OLS(stk_returns_day,factor_day)
        results = model.fit()
        T_values[date] = results.tvalues
        t_value = results.tvalues[0]

        Adj_Rvalues[tomorrow] = results.rsquared_adj
        Num_Cap_coef[tomorrow] = np.corrcoef(factor_day,mkt_cap_day)[0,1]
        Factor_Ret[tomorrow] = results.params[0]
    
        ###去残差后的IC
        model = sm.OLS(factor_day,X_df.iloc[:,1:])
        results = model.fit()
        resid_factors = results.resid
        factor_return_df = pd.concat((resid_factors,stk_returns_day),axis=1)
        IC[tomorrow] = factor_return_df.corr("pearson").iloc[0,1]
        Rank_IC[tomorrow] = factor_return_df.corr("spearman").iloc[0,1]
        titles.append(tomorrow.strftime('%Y/%m/%d')+'  T:%.2f IC:%.2f'%(t_value, Rank_IC[tomorrow]))

    T_value_df = pd.DataFrame(T_values).T
    Info_df = pd.DataFrame({'Adj_R':Adj_Rvalues,'Num_Cap_coef':Num_Cap_coef,'IC':IC,'Rank_IC':Rank_IC,'Factor_Ret':Factor_Ret})

    #####RankIC分布图
    Rank_IC = Info_df['Rank_IC']
    T_values = T_value_df.iloc[:,0]


    avg_RankIC.append(np.nanmean(Rank_IC))
    IC_IR.append( np.nanmean(Rank_IC)/np.nanstd(Rank_IC))
    abs_T_greater2.append( np.nansum(abs(T_values)>2)/len(T_values) )
    factor_return.append( np.nanmean(Info_df['Factor_Ret']) )

    dates = df.index
    df = df.loc[:, STOCK_POOL]
    # print(backtest_factor)
    # std_index_ret = pd.read_csv('utils/std_index_ret.csv',index_col=0, parse_dates=True)
    positions_info,draw_down,start_idx,end_idx = backtest_tool(df, STK_INFO, MKT_CAP, std_index_ret, factor_type=factor_type)
    portfolio_netvalue_df = pd.read_csv('utils/'+factor_type+'portfolio_netvalue.csv',index_col=0, parse_dates=True)
    port_summary = pd.read_csv('utils/'+factor_type+'strategy_summary.csv',index_col=0, encoding='gbk')    
    port_return.append(port_summary.iloc[:5,0].tolist())

port_return = pd.DataFrame(port_return, index=params)
port_return.to_csv('utils/sensitive/%s_partLen_bsns.csv' % factor_type)
regress = pd.DataFrame({'avgIC':np.array(avg_RankIC)*100, 'IC_IR':IC_IR,'abs_T_greater2':abs_T_greater2, 'factor_return':np.array(factor_return)*100}, index=params)
regress.to_csv('utils/sensitive/%s_partLen_rsns.csv' % factor_type)


#######股票池选取敏感性分析
avg_RankIC = []
IC_IR = []
abs_T_greater2 = []
factor_return = []
port_return = []
params = [300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 1800]
for param in params:
    generate_factor_tool(param,10,factor_type=factor_type)
    DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)
    STOCK_POOL = DAILY_FACTOR_DF.columns.tolist()
    # MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()
    decay_window = 12
    decay_factor = 2
    time_std_window = 12
    delay_len = 2

    df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum()
    ##衰减
    df = df.rolling(decay_window).apply(decay, args=(decay_factor,))
    ## 时序标准化
    df = df.rolling(time_std_window).apply(lambda x: 0 if np.nanstd(x)==0 else (x[-1]-np.nanmean(x))/np.nanstd(x))
    ######此时的因子覆盖率
    cover_rate = df.apply(lambda x: np.sum(x!=0)/len(x), axis=1)
    ##缩极值
    df = df.apply(truncate,axis=1,result_type='broadcast')

    ##横截面标准化
    df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')   
    stk_ret = STK_RET.reindex(columns=df.columns)
    df = df + stk_ret - stk_ret
    df = df['20130101':'20180701']

    # 先进行回归
    industry_dummy = pd.DataFrame(np.zeros((len(df.columns),len(industry_list))),index=df.columns,columns=industry_list)
    ####保证stk_info全面性
    for stk in industry_dummy.index:
        industry_dummy.loc[stk, STK_INFO.loc[stk,'industry_sw']] = 1
    industry_dummy = industry_dummy.drop(columns=industry_list[-1])
    
    ########每个月度做回归，并且画出来月度的因子散点图来看
    T_values = {}
    Adj_Rvalues = {}
    Num_Cap_coef = {}
    Factor_Ret = {}
    IC = {}
    Rank_IC = {}
    factor_scatter_list = []
    titles = []
    dates = df.index
    for date_idx in range(len(dates)-1):
        date = dates[date_idx]
        tomorrow = dates[date_idx+1]
        trading_stks = df.columns[~pd.isnull(df.loc[date])]
        stk_returns_day = stk_ret.loc[tomorrow,trading_stks]*0.01
        factor_day = df.loc[date,trading_stks].rename("num_factor")
        factor_scatter_list.append(pd.DataFrame({'factor':factor_day, 'return':stk_returns_day}).values.tolist())
        
        mkt_cap_day = MKT_CAP_FACTOR.loc[date,trading_stks].rename("mkt_cap")
        industry_day = industry_dummy.loc[trading_stks]
        X_df = pd.concat((factor_day,mkt_cap_day,industry_day),axis=1)
        model = sm.OLS(stk_returns_day,X_df)
        # model = sm.OLS(stk_returns_day,factor_day)
        results = model.fit()
        T_values[date] = results.tvalues
        t_value = results.tvalues[0]

        Adj_Rvalues[tomorrow] = results.rsquared_adj
        Num_Cap_coef[tomorrow] = np.corrcoef(factor_day,mkt_cap_day)[0,1]
        Factor_Ret[tomorrow] = results.params[0]
    
        ###去残差后的IC
        model = sm.OLS(factor_day,X_df.iloc[:,1:])
        results = model.fit()
        resid_factors = results.resid
        factor_return_df = pd.concat((resid_factors,stk_returns_day),axis=1)
        IC[tomorrow] = factor_return_df.corr("pearson").iloc[0,1]
        Rank_IC[tomorrow] = factor_return_df.corr("spearman").iloc[0,1]
        titles.append(tomorrow.strftime('%Y/%m/%d')+'  T:%.2f IC:%.2f'%(t_value, Rank_IC[tomorrow]))

    T_value_df = pd.DataFrame(T_values).T
    Info_df = pd.DataFrame({'Adj_R':Adj_Rvalues,'Num_Cap_coef':Num_Cap_coef,'IC':IC,'Rank_IC':Rank_IC,'Factor_Ret':Factor_Ret})

    #####RankIC分布图
    Rank_IC = Info_df['Rank_IC']
    T_values = T_value_df.iloc[:,0]


    avg_RankIC.append(np.nanmean(Rank_IC))
    IC_IR.append( np.nanmean(Rank_IC)/np.nanstd(Rank_IC))
    abs_T_greater2.append( np.nansum(abs(T_values)>2)/len(T_values) )
    factor_return.append( np.nanmean(Info_df['Factor_Ret']) )

    dates = df.index
    df = df.loc[:, STOCK_POOL]
    # print(backtest_factor)
    # std_index_ret = pd.read_csv('utils/std_index_ret.csv',index_col=0, parse_dates=True)
    positions_info,draw_down,start_idx,end_idx = backtest_tool(df, STK_INFO, MKT_CAP, std_index_ret, factor_type=factor_type)
    portfolio_netvalue_df = pd.read_csv('utils/'+factor_type+'portfolio_netvalue.csv',index_col=0, parse_dates=True)
    port_summary = pd.read_csv('utils/'+factor_type+'strategy_summary.csv',index_col=0, encoding='gbk')    
    port_return.append(port_summary.iloc[:5,0].tolist())

port_return = pd.DataFrame(port_return, index=params)
port_return.to_csv('utils/sensitive/%s_stock_pool_bsns.csv' % factor_type)
regress = pd.DataFrame({'avgIC':np.array(avg_RankIC)*100, 'IC_IR':IC_IR,'abs_T_greater2':abs_T_greater2, 'factor_return':np.array(factor_return)*100}, index=params)
regress.to_csv('utils/sensitive/%s_stock_pool_rsns.csv' % factor_type)


#######衰减窗口
generate_factor_tool(600,10,factor_type=factor_type)
avg_RankIC = []
IC_IR = []
abs_T_greater2 = []
factor_return = []
port_return = []
params = [0.25,0.5,0.75,1,1.5,2,3,6,8]
for param in params:
    DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)
    STOCK_POOL = DAILY_FACTOR_DF.columns.tolist()
    # MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()
    decay_window = 12
    decay_factor = param
    time_std_window = 12
    delay_len = 2

    df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum()
    ##衰减
    df = df.rolling(decay_window).apply(decay, args=(decay_factor,))
    ## 时序标准化
    df = df.rolling(time_std_window).apply(lambda x: 0 if np.nanstd(x)==0 else (x[-1]-np.nanmean(x))/np.nanstd(x))
    ######此时的因子覆盖率
    cover_rate = df.apply(lambda x: np.sum(x!=0)/len(x), axis=1)
    ##缩极值
    df = df.apply(truncate,axis=1,result_type='broadcast')

    ##横截面标准化
    df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')   
    stk_ret = STK_RET.reindex(columns=df.columns)
    df = df + stk_ret - stk_ret
    df = df['20130101':'20180701']

    # 先进行回归
    industry_dummy = pd.DataFrame(np.zeros((len(df.columns),len(industry_list))),index=df.columns,columns=industry_list)
    ####保证stk_info全面性
    for stk in industry_dummy.index:
        industry_dummy.loc[stk, STK_INFO.loc[stk,'industry_sw']] = 1
    industry_dummy = industry_dummy.drop(columns=industry_list[-1])
    
    ########每个月度做回归，并且画出来月度的因子散点图来看
    T_values = {}
    Adj_Rvalues = {}
    Num_Cap_coef = {}
    Factor_Ret = {}
    IC = {}
    Rank_IC = {}
    factor_scatter_list = []
    titles = []
    dates = df.index
    for date_idx in range(len(dates)-1):
        date = dates[date_idx]
        tomorrow = dates[date_idx+1]
        trading_stks = df.columns[~pd.isnull(df.loc[date])]
        stk_returns_day = stk_ret.loc[tomorrow,trading_stks]*0.01
        factor_day = df.loc[date,trading_stks].rename("num_factor")
        factor_scatter_list.append(pd.DataFrame({'factor':factor_day, 'return':stk_returns_day}).values.tolist())
        
        mkt_cap_day = MKT_CAP_FACTOR.loc[date,trading_stks].rename("mkt_cap")
        industry_day = industry_dummy.loc[trading_stks]
        X_df = pd.concat((factor_day,mkt_cap_day,industry_day),axis=1)
        model = sm.OLS(stk_returns_day,X_df)
        # model = sm.OLS(stk_returns_day,factor_day)
        results = model.fit()
        T_values[date] = results.tvalues
        t_value = results.tvalues[0]

        Adj_Rvalues[tomorrow] = results.rsquared_adj
        Num_Cap_coef[tomorrow] = np.corrcoef(factor_day,mkt_cap_day)[0,1]
        Factor_Ret[tomorrow] = results.params[0]
    
        ###去残差后的IC
        model = sm.OLS(factor_day,X_df.iloc[:,1:])
        results = model.fit()
        resid_factors = results.resid
        factor_return_df = pd.concat((resid_factors,stk_returns_day),axis=1)
        IC[tomorrow] = factor_return_df.corr("pearson").iloc[0,1]
        Rank_IC[tomorrow] = factor_return_df.corr("spearman").iloc[0,1]
        titles.append(tomorrow.strftime('%Y/%m/%d')+'  T:%.2f IC:%.2f'%(t_value, Rank_IC[tomorrow]))

    T_value_df = pd.DataFrame(T_values).T
    Info_df = pd.DataFrame({'Adj_R':Adj_Rvalues,'Num_Cap_coef':Num_Cap_coef,'IC':IC,'Rank_IC':Rank_IC,'Factor_Ret':Factor_Ret})

    #####RankIC分布图
    Rank_IC = Info_df['Rank_IC']
    T_values = T_value_df.iloc[:,0]


    avg_RankIC.append(np.nanmean(Rank_IC))
    IC_IR.append( np.nanmean(Rank_IC)/np.nanstd(Rank_IC))
    abs_T_greater2.append( np.nansum(abs(T_values)>2)/len(T_values) )
    factor_return.append( np.nanmean(Info_df['Factor_Ret']) )

    dates = df.index
    df = df.loc[:, STOCK_POOL]
    # print(backtest_factor)
    # std_index_ret = pd.read_csv('utils/std_index_ret.csv',index_col=0, parse_dates=True)
    positions_info,draw_down,start_idx,end_idx = backtest_tool(df, STK_INFO, MKT_CAP, std_index_ret, factor_type=factor_type)
    portfolio_netvalue_df = pd.read_csv('utils/'+factor_type+'portfolio_netvalue.csv',index_col=0, parse_dates=True)
    port_summary = pd.read_csv('utils/'+factor_type+'strategy_summary.csv',index_col=0, encoding='gbk')    
    port_return.append(port_summary.iloc[:5,0].tolist())

port_return = pd.DataFrame(port_return, index=params)
port_return.to_csv('utils/sensitive/%s_decayLen_bsns.csv' % factor_type)
regress = pd.DataFrame({'avgIC':np.array(avg_RankIC)*100, 'IC_IR':IC_IR,'abs_T_greater2':abs_T_greater2, 'factor_return':np.array(factor_return)*100}, index=params)
regress.to_csv('utils/sensitive/%s_decayLen_rsns.csv' % factor_type)


#######公告延时窗口
avg_RankIC = []
IC_IR = []
port_return = []
abs_T_greater2 = []
factor_return = []
params = [0,2,5,8,10,15,20,30,45,50]
for param in params:
	# generate_factor_tool(1000,param,factor_type=factor_type)
    DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)
    STOCK_POOL = DAILY_FACTOR_DF.columns.tolist()
    # MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()
    decay_window = 12
    decay_factor = 2
    time_std_window = 12
    delay_len = param

    df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum()
    ##衰减
    df = df.rolling(decay_window).apply(decay, args=(decay_factor,))
    ## 时序标准化
    df = df.rolling(time_std_window).apply(lambda x: 0 if np.nanstd(x)==0 else (x[-1]-np.nanmean(x))/np.nanstd(x))
    ######此时的因子覆盖率
    cover_rate = df.apply(lambda x: np.sum(x!=0)/len(x), axis=1)
    ##缩极值
    df = df.apply(truncate,axis=1,result_type='broadcast')

    ##横截面标准化
    df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')   
    stk_ret = STK_RET.reindex(columns=df.columns)
    df = df + stk_ret - stk_ret
    df = df['20130101':'20180701']

    # 先进行回归
    industry_dummy = pd.DataFrame(np.zeros((len(df.columns),len(industry_list))),index=df.columns,columns=industry_list)
    ####保证stk_info全面性
    for stk in industry_dummy.index:
        industry_dummy.loc[stk, STK_INFO.loc[stk,'industry_sw']] = 1
    industry_dummy = industry_dummy.drop(columns=industry_list[-1])
    
    ########每个月度做回归，并且画出来月度的因子散点图来看
    T_values = {}
    Adj_Rvalues = {}
    Num_Cap_coef = {}
    Factor_Ret = {}
    IC = {}
    Rank_IC = {}
    factor_scatter_list = []
    titles = []
    dates = df.index
    for date_idx in range(len(dates)-1):
        date = dates[date_idx]
        tomorrow = dates[date_idx+1]
        trading_stks = df.columns[~pd.isnull(df.loc[date])]
        stk_returns_day = stk_ret.loc[tomorrow,trading_stks]*0.01
        factor_day = df.loc[date,trading_stks].rename("num_factor")
        factor_scatter_list.append(pd.DataFrame({'factor':factor_day, 'return':stk_returns_day}).values.tolist())
        
        mkt_cap_day = MKT_CAP_FACTOR.loc[date,trading_stks].rename("mkt_cap")
        industry_day = industry_dummy.loc[trading_stks]
        X_df = pd.concat((factor_day,mkt_cap_day,industry_day),axis=1)
        model = sm.OLS(stk_returns_day,X_df)
        # model = sm.OLS(stk_returns_day,factor_day)
        results = model.fit()
        T_values[date] = results.tvalues
        t_value = results.tvalues[0]

        Adj_Rvalues[tomorrow] = results.rsquared_adj
        Num_Cap_coef[tomorrow] = np.corrcoef(factor_day,mkt_cap_day)[0,1]
        Factor_Ret[tomorrow] = results.params[0]
    
        ###去残差后的IC
        model = sm.OLS(factor_day,X_df.iloc[:,1:])
        results = model.fit()
        resid_factors = results.resid
        factor_return_df = pd.concat((resid_factors,stk_returns_day),axis=1)
        IC[tomorrow] = factor_return_df.corr("pearson").iloc[0,1]
        Rank_IC[tomorrow] = factor_return_df.corr("spearman").iloc[0,1]
        titles.append(tomorrow.strftime('%Y/%m/%d')+'  T:%.2f IC:%.2f'%(t_value, Rank_IC[tomorrow]))

    T_value_df = pd.DataFrame(T_values).T
    Info_df = pd.DataFrame({'Adj_R':Adj_Rvalues,'Num_Cap_coef':Num_Cap_coef,'IC':IC,'Rank_IC':Rank_IC,'Factor_Ret':Factor_Ret})

    #####RankIC分布图
    Rank_IC = Info_df['Rank_IC']
    T_values = T_value_df.iloc[:,0]


    avg_RankIC.append(np.nanmean(Rank_IC))
    IC_IR.append( np.nanmean(Rank_IC)/np.nanstd(Rank_IC))
    abs_T_greater2.append( np.nansum(abs(T_values)>2)/len(T_values) )
    factor_return.append( np.nanmean(Info_df['Factor_Ret']) )

    dates = df.index
    df = df.loc[:, STOCK_POOL]
    # print(backtest_factor)
    # std_index_ret = pd.read_csv('utils/std_index_ret.csv',index_col=0, parse_dates=True)
    positions_info,draw_down,start_idx,end_idx = backtest_tool(df, STK_INFO, MKT_CAP, std_index_ret, factor_type=factor_type)
    portfolio_netvalue_df = pd.read_csv('utils/'+factor_type+'portfolio_netvalue.csv',index_col=0, parse_dates=True)
    port_summary = pd.read_csv('utils/'+factor_type+'strategy_summary.csv',index_col=0, encoding='gbk')    
    port_return.append(port_summary.iloc[:5,0].tolist())

port_return = pd.DataFrame(port_return, index=params)
port_return.to_csv('utils/sensitive/%s_delayLen_bsns.csv' % factor_type)
delay_len = pd.DataFrame({'avgIC':np.array(avg_RankIC)*100, 'IC_IR':IC_IR,'abs_T_greater2':abs_T_greater2, 'factor_return':np.array(factor_return)*100}, index=params)
delay_len.to_csv('utils/sensitive/%s_delayLen_rsns.csv' % factor_type)



############买方卖方的区别

# avg_RankIC = []
# IC_IR = []
# port_return = []
# abs_T_greater2 = []
# factor_return = []
# params = ['both','buy','sell']
# for param in params:
#     generate_factor_tool(600,10,side=param,factor_type=factor_type)
#    # MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()
#     DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)
#     STOCK_POOL = DAILY_FACTOR_DF.columns.tolist()
#     # MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()
#     decay_window = 12
#     decay_factor = 2
#     time_std_window = 12
#     delay_len = 2

#     df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum()
#     ##衰减
#     df = df.rolling(decay_window).apply(decay, args=(decay_factor,))
#     ## 时序标准化
#     df = df.rolling(time_std_window).apply(lambda x: 0 if np.nanstd(x)==0 else (x[-1]-np.nanmean(x))/np.nanstd(x))
#     ######此时的因子覆盖率
#     cover_rate = df.apply(lambda x: np.sum(x!=0)/len(x), axis=1)
#     ##缩极值
#     df = df.apply(truncate,axis=1,result_type='broadcast')

#     ##横截面标准化
#     df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')   
#     stk_ret = STK_RET.reindex(columns=df.columns)
#     df = df + stk_ret - stk_ret
#     df = df['20130101':'20180701']

#     # 先进行回归
#     industry_dummy = pd.DataFrame(np.zeros((len(df.columns),len(industry_list))),index=df.columns,columns=industry_list)
#     ####保证stk_info全面性
#     for stk in industry_dummy.index:
#         industry_dummy.loc[stk, STK_INFO.loc[stk,'industry_sw']] = 1
#     industry_dummy = industry_dummy.drop(columns=industry_list[-1])
    
#     ########每个月度做回归，并且画出来月度的因子散点图来看
#     T_values = {}
#     Adj_Rvalues = {}
#     Num_Cap_coef = {}
#     Factor_Ret = {}
#     IC = {}
#     Rank_IC = {}
#     factor_scatter_list = []
#     titles = []
#     dates = df.index
#     for date_idx in range(len(dates)-1):
#         date = dates[date_idx]
#         tomorrow = dates[date_idx+1]
#         trading_stks = df.columns[~pd.isnull(df.loc[date])]
#         stk_returns_day = stk_ret.loc[tomorrow,trading_stks]*0.01
#         factor_day = df.loc[date,trading_stks].rename("num_factor")
#         factor_scatter_list.append(pd.DataFrame({'factor':factor_day, 'return':stk_returns_day}).values.tolist())
        
#         mkt_cap_day = MKT_CAP_FACTOR.loc[date,trading_stks].rename("mkt_cap")
#         industry_day = industry_dummy.loc[trading_stks]
#         X_df = pd.concat((factor_day,mkt_cap_day,industry_day),axis=1)
#         model = sm.OLS(stk_returns_day,X_df)
#         # model = sm.OLS(stk_returns_day,factor_day)
#         results = model.fit()
#         T_values[date] = results.tvalues
#         t_value = results.tvalues[0]

#         Adj_Rvalues[tomorrow] = results.rsquared_adj
#         Num_Cap_coef[tomorrow] = np.corrcoef(factor_day,mkt_cap_day)[0,1]
#         Factor_Ret[tomorrow] = results.params[0]
    
#         ###去残差后的IC
#         model = sm.OLS(factor_day,X_df.iloc[:,1:])
#         results = model.fit()
#         resid_factors = results.resid
#         factor_return_df = pd.concat((resid_factors,stk_returns_day),axis=1)
#         IC[tomorrow] = factor_return_df.corr("pearson").iloc[0,1]
#         Rank_IC[tomorrow] = factor_return_df.corr("spearman").iloc[0,1]
#         titles.append(tomorrow.strftime('%Y/%m/%d')+'  T:%.2f IC:%.2f'%(t_value, Rank_IC[tomorrow]))

#     T_value_df = pd.DataFrame(T_values).T
#     Info_df = pd.DataFrame({'Adj_R':Adj_Rvalues,'Num_Cap_coef':Num_Cap_coef,'IC':IC,'Rank_IC':Rank_IC,'Factor_Ret':Factor_Ret})

#     #####RankIC分布图
#     Rank_IC = Info_df['Rank_IC']
#     T_values = T_value_df.iloc[:,0]


#     avg_RankIC.append(np.nanmean(Rank_IC))
#     IC_IR.append( np.nanmean(Rank_IC)/np.nanstd(Rank_IC))
#     abs_T_greater2.append( np.nansum(abs(T_values)>2)/len(T_values) )
#     factor_return.append( np.nanmean(Info_df['Factor_Ret']) )

#     dates = df.index
#     df = df.loc[:, STOCK_POOL]
#     # print(backtest_factor)
#     # std_index_ret = pd.read_csv('utils/std_index_ret.csv',index_col=0, parse_dates=True)

#     positions_info,draw_down,start_idx,end_idx = backtest_tool(df, STK_INFO, MKT_CAP, std_index_ret)
#     port_summary = pd.read_csv('utils/'+factor_type+'strategy_summary.csv',index_col=0, encoding='gbk')
#     port_summary.to_csv('utils/'+param+'strategy_summary.csv')
    # port_return.append(port_summary.iloc[:5,0].tolist())

# port_return = pd.DataFrame(port_return, index=params)
# port_return.to_csv('utils/sensitive/delayLen_bsns.csv')
# delay_len = pd.DataFrame({'avgIC':np.array(avg_RankIC)*100, 'IC_IR':IC_IR,'abs_T_greater2':abs_T_greater2, 'factor_return':np.array(factor_return)*100}, index=params)
# delay_len.to_csv('utils/sensitive/delayLen_rsns.csv')