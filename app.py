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
app.config['SECRET_KEY'] = "IM A CUTE BOY"
bootstrap = Bootstrap(app)
client = MongoClient()
collection = client.research.newTable

global PRESENT_MODE
PRESENT_MODE = True

global IPV4
IPV4 = "120.79.242.98:5000"

global START 
START = '20130101'
global END
END = '20180701'

## industry_list
global INDUSTRY_LIST
INDUSTRY_LIST = collection.find({}).distinct("stkIndustry")
global STK_INFO
global STK_INFO2
STK_INFO = pd.read_excel('utils/股票信息词典.xlsx',encoding='gbk',index_col=2)
STK_INFO2 =pd.read_excel('utils/股票信息词典.xlsx',encoding='gbk', index_col=0)

factor_type = 'partNum'
global DAILY_FACTOR_DF
DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)

global MONTHLY_FACTOR_DF
MONTHLY_FACTOR_DF = pd.read_csv('utils/factor_monthly_raw.csv',index_col=0,parse_dates = True)

global MKT_CAP
MKT_CAP = pd.read_csv("utils/monthly_cap.csv",index_col=0,parse_dates = True)[START:END]

global STK_RET
STK_RET = pd.read_csv("utils/monthly_return.csv",index_col=0,parse_dates = True)

global MOMENTUM
MOMENTUM = pd.read_csv('utils/other_factors/momentum3M.csv',index_col=0, parse_dates=True)[START:END]

global PB
PB = pd.read_csv('utils/other_factors/monthly_pb.csv',index_col=0, parse_dates=True)[START:END]

global TURN_OVER
TURN_OVER = pd.read_csv('utils/other_factors/monthly_turnover.csv',index_col=0, parse_dates=True)[START:END]

global FACTOR_TYPE
FACTOR_TYPE = 'partNum'

global MKT_CAP_FACTOR
df = np.log(MKT_CAP).apply(truncate,axis=1,result_type='broadcast')
df = df.apply(lambda x: 0 if np.nanstd(x)==0 else (x-np.nanmean(x))/np.nanstd(x),axis=1,result_type='broadcast')
MKT_CAP_FACTOR = df

global SIDE_FILTER
SIDE_FILTER = 'both'

global ZZ500 
ZZ500 = pd.read_csv('utils/other_factors/zz500.csv', index_col=0).index.tolist()

global STOCK_POOL
STOCK_POOL = [0]*600
#############自然语言处理
global STOP_WORDS
material_path = 'utils/nlp/'
with open(material_path+'my_stopwords.txt', 'r', encoding='utf-8') as f:
	stopwords = [line.strip() for line in f.readlines()]
STOP_WORDS = stopwords
MY_DICT = material_path+'build_mydict.txt'
jieba.load_userdict(MY_DICT)

############### load NLP corpora
dictionary = corpora.Dictionary.load(material_path+'after2015.dict')
# corpus = corpora.mmcorpus.MmCorpus.load(material_path+'data\\after2015corpus.mm')
tfidf_model =  TfidfModel.load(material_path+'after2015_tfidf')

# # collection = client.invAct.invActivity
# factor_types = ['partNum','OrgNum','QuestionNum','Dummy']
# for factor in factor_types:
#     return_file = 'utils/'+factor+'portfolio_return.csv'
#     ret_df = pd.read_csv(return_file,index_col=0, parse_dates=True)
#     net_value_df = get_netvalue(ret_df)
#     years = [str(year) for year in pd.unique(ret_df.index.year)]
#     year_summarys = {}
#     for year in years:
#
#         summary_df = get_summary(ret_df[year],benchmark='hs300')
#         year_summarys[year] = summary_df.loc[:,'年化收益率']
#     df = pd.DataFrame(year_summarys, index=summary_df.index)
#     df.to_csv('Yearly'+factor+'.csv')
    



@app.route('/', methods=['GET','POST'])
def index():
    return render_template('welcome_page.html',ipv4=IPV4)

@app.route('/outline', methods=['GET','POST'])
def outline():
    return render_template('outline.html')


@app.route('/original_docs', methods=['GET','POST'])
def original_docs():
    form = NameForm()
    # global first_visit_doc
    docs = []
    wordcloud = ''

    if not form.validate_on_submit():
        form.stkCode.data = '紫光股份'
        form.start_date.data = '20170101'
        form.end_date.data = '20180701'

    if form.validate_on_submit():
        stkName = form.stkCode.data
        start_date = datetime.datetime.strptime(form.start_date.data,'%Y%m%d')
        docs = []
        wordcloud = []
        if form.end_date.data:
            end_date = datetime.datetime.strptime(form.end_date.data, '%Y%m%d')
        else:
            end_date = start_date
        if stkName in INDUSTRY_LIST:
            # cursor = collection.find({"stkIndustry": stkName, "statementNum": {"$gt": 20}})
            cursor = collection.find({"activityDate": {"$gte":start_date,"$lte":end_date},"statementNum":{"$gte":20}, "stkIndustry": stkName})
        else:
            cursor = collection.find({"stkName": stkName, "activityDate": {"$gte": start_date, "$lte": end_date}})
        for doc in cursor:
            doc['activityDate'] = datetime.datetime.strftime(doc['activityDate'], '%Y-%m-%d')
            doc['announceDate'] = datetime.datetime.strftime(doc['announceDate'], '%Y-%m-%d')
            words, html = sent_analyse(doc['content'])
            wordcloud.extend(words)
            doc['cutWords'] = html #if word not in STOP_WORDS]
            docs.append(doc)
        stk_vector = tfidf_model[dictionary.doc2bow(wordcloud)]
        words =[]
        weights = []
        for idx,weight in stk_vector:
            words.append(dictionary.id2token[idx])
            weights.append(weight)
        # print(words)
        wordcloud = WordCloud("云图",width=800,height=400)
        wordcloud.add("",words,weights)
        wordcloud = wordcloud.render_embed()

    return render_template('queryDb.html', form=form,docs=docs,wordcloud=wordcloud,ipv4=IPV4)

@app.route('/db_type_summary', methods=['GET','POST'])
def db_type_summary():
    html = 'myhtml/各调研类别信息详解2013.html'
    soup = BeautifulSoup(open(html,'r',encoding='utf-8'), 'html.parser')
    mychart = str(soup.body)
    return render_template('type_summary.html', mychart=mychart, ipv4=IPV4)

@app.route('/sent_demo', methods=['GET','POST'])
def sent_demo():
    return render_template('sent_demo.html',ipv4=IPV4)

@app.route('/sent_demo_query', methods=['GET','POST'])
def sent_demo_query():
    _,html = sent_analyse(request.form['input'])
    return json.dumps({'output':html})



@app.route('/get_scatter', methods=['GET','POST'])
def get_scatter():
    response = {}
    response['dates'] = [str(i) for i in range(60)]
    response['datas'] = np.random.random((60,100,2)).tolist()
    return json.dumps(response)


@app.route('/single_factor_backtest', methods=['GET','POST'])
def single_factor_backtest():
    return render_template('single_factor_backtest.html',ipv4=IPV4)

@app.route('/generate_factor', methods=['GET','POST'])
def generate_factor():
    stk_pool = int(request.form['stk_pool'])
    side_filter = request.form['side_filter']
    type_filter = request.form.getlist(key='type_filter[]')
    partLen_thd = int(request.form['partLen_thd'])
    factor_type = request.form['factor_type']
    generate_factor_tool(stk_pool,partLen_thd,side_filter,type_filter,factor_type= factor_type)
    global DAILY_FACTOR_DF
    DAILY_FACTOR_DF = pd.read_csv('utils/' + factor_type + 'factor_daily_raw.csv', index_col=0, parse_dates=True)
    global MONTHLY_FACTOR_DF
    MONTHLY_FACTOR_DF = DAILY_FACTOR_DF.resample('M').sum()[START:END]
    global STOCK_POOL
    STOCK_POOL = MONTHLY_FACTOR_DF.columns.tolist()

    return json.dumps({})



@app.route('/update_draw',methods=['GET','POST'])
def update_draw():

    stock_pool = int(request.form["stock_pool"])
    date_range = request.form["time_range"].split(';')
    global MONTHLY_FACTOR_DF
    DAILY_FACTOR_DF = pd.read_csv('utils/'+factor_type+'factor_daily_raw.csv',index_col=0,parse_dates = True)
    factor_df = DAILY_FACTOR_DF.resample('M').sum().loc[date_range[0]:date_range[1],:]
    factor_df = factor_df.reindex(columns=factor_df.sum().sort_values(ascending=False).index).iloc[:,:stock_pool]
    stocks = factor_df.columns

    industry_dist = pd.value_counts(STK_INFO.loc[stocks,'industry_sw'])
    zz500_dist = pd.value_counts(STK_INFO.loc[ZZ500,'industry_sw'])
    mkt_cap = MKT_CAP.iloc[-1,:].loc[stocks] / 100000000 #以亿记数
    hist = np.histogram(mkt_cap,bins=[0,1,5,10,15,20,30,40,50,80,100,200,300,500,800,1000,3000,5000])
    hist_x = [str(item)+'亿' for item in hist[1][:-1]]
    hist_y = hist[0].astype(np.float).tolist()

    cover_rate = factor_df.apply(lambda x: np.sum(x != 0) / len(x), axis=1)

    response = {\
        "industry": [{"name":industry_dist.index[i],"value":int(industry_dist.iloc[i])} for i in range(len(industry_dist))],
        "zz500_dist":[{"name":zz500_dist.index[i],"value":int(zz500_dist.iloc[i])} for i in range(len(zz500_dist))],
        "cover_rate_x": [stamp.strftime('%Y/%m/%d') for stamp in cover_rate.index],
        "cover_rate_y": cover_rate.tolist(),
        "mkt_hist_x": hist_x,
        "mkt_hist_y": hist_y,

        }

    return json.dumps(response)

@app.route('/process_factor', methods=['GET','POST'])
def process_factor():

    global FACTOR_TYPE
    FACTOR_TYPE = request.form['factor_type']
    decay_window = int(request.form["decay_window"])
    decay_factor = float(request.form["decay_factor"])
    time_std_window = int(request.form["time_std_window"])
    delay_len = int(request.form["delay_len"])
    global SIDE_FILTER
    SIDE_FILTER = request.form["side_filter"]

    if PRESENT_MODE:
        tmp = FACTOR_TYPE
        if tmp in ['turnover', 'pb', 'momentum']:
            tmp = 'partNum'
        json_path = 'results/process_%d_%s_%s' % (len(STOCK_POOL), tmp, SIDE_FILTER)
        with open(json_path, 'r') as f:
            response = json.load(f)
        return json.dumps(response)
    else:

        global DAILY_FACTOR_DF
        DAILY_FACTOR_DF = pd.read_csv('utils/'+FACTOR_TYPE+'factor_daily_raw.csv', index_col=0, parse_dates=True)
        df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum().loc[:,STOCK_POOL]
        df = df[START:END]
        # 数量因子在每个分组上求和 看看差别
        
        ##几个因子间的相关性
        size = MKT_CAP_FACTOR.loc[:,STOCK_POOL].fillna(0)
        pb = PB.loc[:,STOCK_POOL].fillna(0)
        momentum = MOMENTUM.loc[:,STOCK_POOL].fillna(0)
        turnover = TURN_OVER.loc[:,STOCK_POOL].fillna(0)
        
        rank_info = pd.DataFrame(np.zeros((4,10)),columns=list(range(10)), index=['size','pb','momentum','turnover'])
        gap = len(STOCK_POOL) // 10
        cutPoint = [gap*i for i in range(10)]
        cutPoint.append(len(STOCK_POOL))
        
        
        for date in df.index:
            size_info = size.loc[date,:]
            size_info = df.loc[date, size_info.index[np.argsort(size_info)]]
            for i in range(10):
                rank_info.loc['size',i] += np.sum(size_info[cutPoint[i]:cutPoint[i+1]])
            pb_info = pb.loc[date,:]
            pb_info = df.loc[date, pb_info.index[np.argsort(pb_info)]]
            for i in range(10):
                rank_info.loc['pb',i] += np.sum(pb_info[cutPoint[i]:cutPoint[i+1]])
            momentum_info = momentum.loc[date,:]
            momentum_info = df.loc[date, momentum_info.index[np.argsort(momentum_info)]]
            for i in range(10):
                rank_info.loc['momentum',i] += np.sum(momentum_info[cutPoint[i]:cutPoint[i+1]])
            turnover_info = turnover.loc[date,:]
            turnover_info = df.loc[date, turnover_info.index[np.argsort(turnover_info)]]
            for i in range(10):
                rank_info.loc['turnover',i] += np.sum(turnover_info[cutPoint[i]:cutPoint[i+1]])
        
        rank_info = rank_info.apply(lambda x: x/np.sum(x), axis=1)
        corr_series = [{'name':item,'type':'bar','data':rank_info.loc[item].tolist()} for item in rank_info.index]
        corr_series_x = ['低']+[str(i) for i in range(1,9)]+['高']
        
        df = DAILY_FACTOR_DF.shift(delay_len).resample('M').sum().loc[:,STOCK_POOL]
        
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
        df = df[START:END]
        df.to_csv('utils/'+FACTOR_TYPE+'backtest_factor.csv')
        global MONTHLY_FACTOR_DF
        MONTHLY_FACTOR_DF = df
        #
    
        #
        # #######因子的截面相关系数
        tmp = df.fillna(0)
        cross_corr = []
        for i in range(df.shape[0]-1):
            cross_corr.append(np.corrcoef(tmp.iloc[i,:].tolist(), tmp.iloc[i+1,:].tolist())[0,1])
        # cross_corr.append(cross_corr[-1])
        #####因子现在的覆盖度
        factor_cover_after = df.apply(lambda x: 1-np.sum(pd.isnull(x))/df.shape[1])[:-1].tolist()
        
        
        response = {}
        response['dates'] = [stamp.strftime('%Y/%m/%d') for stamp in df.index[:-1]]
        response['cross_corr'] = cross_corr
        response['factor_cover_after'] = factor_cover_after
        response["corr_series"] = corr_series
        response["corr_series_x"] = corr_series_x

        json_path = 'results/process_%d_%s_%s' % (len(STOCK_POOL), FACTOR_TYPE, SIDE_FILTER)
        with open(json_path, 'w') as f:
            json.dump(response, f)
    
    
        return json.dumps(response)



@app.route('/regress', methods=['GET','POST'])
def regress():
    if PRESENT_MODE:

        json_path = 'results/regress_%d_%s_%s' % (len(STOCK_POOL), FACTOR_TYPE, SIDE_FILTER)
        with open(json_path, 'r') as f:
            response = json.load(f)
        return json.dumps(response)

    else:
        df = pd.read_csv('utils/'+FACTOR_TYPE+'backtest_factor.csv', index_col=0, parse_dates=True)
        df = df.loc[START:END, STOCK_POOL]
        industry_list = INDUSTRY_LIST
    
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
        T_value_df.to_csv('utils/'+FACTOR_TYPE+'T_values.csv',encoding='gbk')
        Info_df.to_csv('utils/'+FACTOR_TYPE+'other_info.csv',encoding='gbk')
        
        date_strList = [stamp.strftime('%Y/%m/%d') for stamp in Info_df.index]
        
        #####RankIC分布图
        Rank_IC = Info_df['Rank_IC']
        T_values = T_value_df.iloc[:,0]
        hist = np.histogram(Rank_IC,bins=20)
        IC_hist_x = ['%.2f'%item for item in hist[1][:-1]]
        IC_hist_y = hist[0].astype(np.float).tolist()
        
        response = {}
        response['dates'] = date_strList
        response['Rank_IC'] = Rank_IC.tolist()
        response['T_values'] = T_values.tolist()
        response['IC_hist_x'] = IC_hist_x
        response['IC_hist_y'] = IC_hist_y
        IC_summary = [['avg','std','min','max','IC_IR','IC的T值','IC>0占比'],\
                        [np.nanmean(Rank_IC), np.nanstd(Rank_IC), np.nanmin(Rank_IC), np.nanmax(Rank_IC),\
                         np.nanmean(Rank_IC)/np.nanstd(Rank_IC), np.nanmean(Rank_IC)*np.sqrt(len(Rank_IC)-1)/np.nanstd(Rank_IC),np.nansum(Rank_IC>0)/len(Rank_IC)]]
        T_value_summary = [['avg','std','min','max','T>0占比','|T|>2占比','因子收益率均值'],\
                        [np.nanmean(T_values), np.nanstd(T_values), np.nanmin(T_values), np.nanmax(T_values),\
                         np.nansum(T_values>0)/len(T_values), np.nansum(abs(T_values)>2)/len(T_values), np.nanmean(Info_df['Factor_Ret'])]]
        response['IC_summary_table'] = pd.DataFrame(np.vstack((IC_summary, T_value_summary)).T,columns=['RankIC','值','T-Value','值']).to_html(index=False)
        response['scatter_titles'] = titles
        response['factor_scatter_data'] = factor_scatter_list
    
        json_path = 'results/regress_%d_%s_%s' % (len(STOCK_POOL), FACTOR_TYPE, SIDE_FILTER)
        with open(json_path, 'w') as f:
            json.dump(response, f)
    
        return json.dumps(response)


@app.route('/backtest', methods=['GET','POST'])
def backtest():

    if PRESENT_MODE:

        json_path = 'results/backtest_%d_%s_%s' % (len(STOCK_POOL), FACTOR_TYPE, SIDE_FILTER)
        with open(json_path, 'r') as f:
            response = json.load(f)
        return json.dumps(response)

    else:

        df = pd.read_csv('utils/'+FACTOR_TYPE+'backtest_factor.csv', index_col=0, parse_dates=True)
        df = df.loc[START:END, STOCK_POOL]
        dates = df.index
        # print(backtest_factor)
        std_index_ret = pd.read_csv('utils/std_index_ret.csv',index_col=0, parse_dates=True)
        positions_info,draw_down,start_idx,end_idx = backtest_tool(df, STK_INFO, MKT_CAP, std_index_ret,factor_type=FACTOR_TYPE)
        portfolio_netvalue_df = pd.read_csv('utils/'+FACTOR_TYPE+'portfolio_netvalue.csv',index_col=0, parse_dates=True)
        port_summary = pd.read_csv('utils/'+FACTOR_TYPE+'strategy_summary.csv',index_col=0, encoding='gbk')
        response = {}
        response['dates'] = portfolio_netvalue_df.index.strftime('%Y%m%d').tolist()
        data_series = [{'name':col,"type":"line","showSymbol":0,"data":np.round(portfolio_netvalue_df[col],3).tolist()} for col in portfolio_netvalue_df.columns]
        
        draw_down_series = {
        'name':'draw_down','yAxisIndex':1,'areaStyle':{'normal':{}},'type':'line','data':draw_down,\
        'markArea':{'silent':1, 'data':[[{'xAxis':response['dates'][start_idx]},{'xAxis':response['dates'][end_idx]}]]}
        }
        data_series.append(draw_down_series)
        response['series'] =data_series
        response['series'].append(draw_down_series)
        response['positions_info'] = positions_info
        response['backtest_summary'] = port_summary.to_html()
        columns_tmp = ['年化收益率','夏普比率','最大回撤']
        response['backtest_draw'] = columns_tmp
        response['backtest_draw_series'] = [{'name':port_summary.index[i],'type':'bar','data':port_summary.iloc[i].loc[columns_tmp].tolist()} for i in range(5)]
        
        turnover_info = [[1]*5]
        port1_mktcap = []
        portall_mktcap = []
        
        for i in range(1,len(positions_info)):
            # 当天的各组合换手率
            tmp = []
            for j in range(5):
                previous = positions_info[i-1]['pos'][j]['sec_name']
                today = positions_info[i]['pos'][j]['sec_name']
                tmp.append(cal_turnover(previous,today))
            turnover_info.append(tmp)
            port1_mktcap.append(np.nanmedian(MKT_CAP.loc[dates[i], STK_INFO2.loc[ positions_info[i]['pos'][0]['sec_name'],'wind_code'] ]))
            portall_mktcap.append(np.nanmedian(MKT_CAP.loc[dates[i], STOCK_POOL]))
        
        turnover_info = np.array(turnover_info)
        turnover_series = [{'name':'组合'+str(idx+1),"type":"line","showSymbol":0,"data":turnover_info[:,idx].tolist()} for idx in range(5)]
        response['turnover_series'] = turnover_series
        
        mktcap_series = [{'name':'Port1',"type":"line","data":port1_mktcap}, {'name':'STOCK_POOL',"type":"line", "data":portall_mktcap},\
                        {'name':'ratio',"type":"line","data":(np.array(port1_mktcap)/np.array(portall_mktcap)).tolist(), "yAxisIndex":1}]
        response['mktcap_series'] = mktcap_series
    
        json_path = 'results/backtest_%d_%s_%s' % (len(STOCK_POOL), FACTOR_TYPE, SIDE_FILTER)
        with open(json_path, 'w') as f:
            json.dump(response, f)
       
    

        return json.dumps(response)

@app.route('/sensitive_anaylysis', methods=['GET','POST'])
def sensitive_anaylysis():
    response = {}
    tmp = ''
    global FACTOR_TYPE
    tmp = FACTOR_TYPE
    if FACTOR_TYPE in ['turnover','pb','momentum']:
        FACTOR_TYPE = 'partNum'

    stk_pool_bsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'stock_pool_bsns.csv', index_col=0)
    stk_pool_rsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'stock_pool_rsns.csv', index_col=0)
    response['stk_pool_x'] = list(map(str,stk_pool_bsns.index))
    response['stk_pool_bsns_series'] = [{'name':'Port'+str(i+1), 'type':'bar', 'data': stk_pool_bsns.iloc[:,i].tolist()} for i in range(stk_pool_bsns.shape[1])]
    response['stk_pool_rsns_series'] = [{'name': stk_pool_rsns.columns[i], 'type':'line', 'data': stk_pool_rsns.iloc[:,i].tolist()} for i in range(stk_pool_rsns.shape[1])]


    partLen_bsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'partLen_bsns.csv', index_col=0)
    partLen_rsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'partLen_rsns.csv', index_col=0)
    response['partLen_x'] = list(map(str,partLen_bsns.index))
    response['partLen_bsns_series'] = [{'name':'Port'+str(i+1), 'type':'bar', 'data': partLen_bsns.iloc[:,i].tolist()} for i in range(partLen_bsns.shape[1])]
    response['partLen_rsns_series'] = [{'name': partLen_rsns.columns[i], 'type':'line', 'data': partLen_rsns.iloc[:,i].tolist()} for i in range(partLen_rsns.shape[1])]



    IC_decay_rsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'IC_decay_rsns.csv', index_col=0)
    response['IC_decay_x'] = list(map(str,IC_decay_rsns.index))
    response['IC_decay_rsns_series'] = [{'name': IC_decay_rsns.columns[i], 'type':'line', 'data': IC_decay_rsns.iloc[:,i].tolist()} for i in range(IC_decay_rsns.shape[1])]



    decayLen_bsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'decayLen_bsns.csv', index_col=0)
    decayLen_rsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'decayLen_rsns.csv', index_col=0)
    response['decayLen_x'] = list(map(str,decayLen_bsns.index))
    response['decayLen_bsns_series'] = [{'name':'Port'+str(i+1), 'type':'bar', 'data': decayLen_bsns.iloc[:,i].tolist()} for i in range(decayLen_bsns.shape[1])]
    response['decayLen_rsns_series'] = [{'name': decayLen_rsns.columns[i], 'type':'line', 'data': decayLen_rsns.iloc[:,i].tolist()} for i in range(decayLen_rsns.shape[1])]


    delayLen_bsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'delayLen_bsns.csv', index_col=0)
    delayLen_rsns = pd.read_csv('utils/sensitive/'+FACTOR_TYPE+'_'+'delayLen_rsns.csv', index_col=0)
    response['delayLen_x'] = list(map(str,delayLen_bsns.index))
    response['delayLen_bsns_series'] = [{'name':'Port'+str(i+1), 'type':'bar', 'data': delayLen_bsns.iloc[:,i].tolist()} for i in range(delayLen_bsns.shape[1])]
    response['delayLen_rsns_series'] = [{'name': delayLen_rsns.columns[i], 'type':'line', 'data': delayLen_rsns.iloc[:,i].tolist()} for i in range(delayLen_rsns.shape[1])]
    FACTOR_TYPE = tmp
    return json.dumps(response)

if __name__ == '__main__':
    app.run(host=IPV4[:-5],port=5000,debug=False)
