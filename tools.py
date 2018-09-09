import pandas as pd
import numpy as np
import datetime
import re
import jieba
from wtforms import StringField, SubmitField,TextAreaField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
from pymongo import MongoClient
# from WindPy import w
from collections import defaultdict
# w.start()
client = MongoClient()
collection = client.research.newTable
industry_list = collection.find({}).distinct("stkIndustry")

material_path = 'utils/nlp/'
with open('utils/nlp/my_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f.readlines()]
my_sent_df = pd.read_csv(material_path+'sent_df.csv',encoding='gb18030')
my_sent_df = my_sent_df.drop_duplicates('word',keep='last').set_index('word')
positive_df = my_sent_df[my_sent_df['score']>0]
negative_df = my_sent_df[my_sent_df['score']<0]

most_degree = ['百分之百', '倍加','全部','全面', '备至', '不得了', '不堪', '不可开交', '不亦乐乎', '不折不扣', '彻头彻尾', '充分', '到头', '地地道道', '非常', '极', '极度', '极其', '极为', '截然', '尽', '惊人地', '绝', '绝顶', '绝对', '绝对化', '刻骨', '酷', '满', '满贯', '满心', '莫大', '奇', '入骨', '甚为', '十二分', '十分', '十足', '死', '滔天', '痛', '透', '完全', '完完全全', '万', '万般', '万分', '万万', '无比', '无度', '无可估量', '无以复加', '无以伦比', '要命', '要死', '已极', '已甚', '异常', '逾常', '贼', '之极', '之至', '至极', '卓绝', '最为', '佼佼', '郅', '綦', '齁', '最']

very_degree = ['非常',"大幅",'迅速','快速','大幅度','相当','很','重大','不过', '不少', '不胜', '惨', '沉', '沉沉', '出奇', '大为', '多', '多多', '多加', '多么', '分外', '格外', '够瞧的', '够戗', '好', '好不', '何等', '很', '很是', '坏', '可', '老', '老大', '良', '颇', '颇为', '甚', '实在', '太', '太甚', '特', '特别', '尤', '尤其', '尤为', '尤以', '远', '着实', '曷', '碜']

more_degree = ['大不了', '多', '大','更', '更加', '更进一步', '更为', '还', '还要', '较', '较比', '较大','较为', '进一步', '那般', '那么', '那样', '强', '如斯', '益', '益发', '尤甚', '逾', '愈', '愈发', '愈加', '愈来愈', '愈益', '远远',  '越发', '越加', '越来越', '越是', '这般', '这样', '足', '足足']

ish_degree = ['点点滴滴', '多多少少', '小幅','怪', '好生', '还', '或多或少', '略', '略加', '略略', '略微', '略为', '蛮', '稍', '稍稍', '稍微', '稍为', '稍许', '挺', '未免', '相当', '些', '些微', '些小', '一点', '一点儿', '一些', '有点', '有点儿', '有些']

least_degree = ['半点', '轻微','聊',  '轻度', '弱', '丝毫', '微', '相对','倒是']

neg_degree = ['无','不','没有','防控','防止','不是','不为过','不够','避免', '超','不甚', '不怎么','没怎么',  '超外差',  '从不','从严','决不','绝非','决非', '从未','过度', '过分', '过火', '过劲', '过了头', '过猛', '过热', '过甚', '过头', '过于', '过逾', '何止', '何啻', '开外', '苦', '老',  '强', '溢', '忒']



############filter

buyside_filter = ["基金管理公司","投资公司","资产管理公司","保险资产管理公司","寿险公司","信托公司","财险公司"]
sellside_filter = ["证券公司"]
bothside_filter = buyside_filter+sellside_filter
specific = ["特定对象调研"]
presence = ["现场参观","分析师会议","路演活动","-"]
others = ["其他","电话沟通","电话会议","业绩说明会","媒体采访","投资者接待日"]

def split_sentences(raw):
    pattern = re.compile(r'(问题[一二三四五六七八九十\d]|\d+、|问：|[一二三四五六七八九十]、|\d+[\.\．](?!\d)|Q:|Q：|（\d+）)')
    matches = list(re.finditer(pattern, raw))
    if len(matches) == 0:
        return ([raw],0)
    else:
        sentences = []
        # sentences.append(raw[:matches[0].span()[0]])
        for idx,match in enumerate(matches[:-1]):
            sentences.append(raw[match.span()[0]: matches[idx+1].span()[0]])
        sentences.append(raw[matches[-1].span()[0]:])
        return(sentences,len(sentences))

def my_partition(stk_num, port_num):
    #'s: stk_num  p:port_num, parition out the portfolios'
    portfolios = []
    # if stk_num==0:
    #     for p in range(port_num):
    #         portseries = pd.DataFrame(None,columns=['stk_idx','weight'])
    #         portseries['stk_idx'] = [0]
    #         portseries['weight'] = [0]
    #         portfolios.append(portseries)            
    if True:
    # if stk_num < port_num:
        idx = 1
        for p in range(port_num):               
            stk_idx = []
            stk_weight = []
            stk_idx.append(idx)
            stk_weight.append(min(1/port_num,idx/stk_num-p/port_num))
            while (idx+1)/stk_num <= (p+1)/port_num:
                stk_idx.append(idx+1)
                stk_weight.append(1/stk_num)
                idx+=1
            if (p+1)/port_num > idx/stk_num:
                stk_idx.append(idx+1)
                stk_weight.append((p+1)/port_num - idx/stk_num)
                idx+=1
            stk_idx = np.array(stk_idx)-1
            weight_value = np.array(stk_weight)/ sum(stk_weight)
            portseries = pd.DataFrame(None,columns=['stk_idx','weight'])
            portseries['stk_idx'] = stk_idx
            portseries['weight'] = weight_value
            portfolios.append(portseries)
    return portfolios

def transfer2df(doc_list):
    res = defaultdict(list)
    for doc in doc_list:
        for key,value in doc.items():
            if type(value) is dict:
                for inner_key,inner_value in value.items():
                    res[inner_key].append(inner_value)
            else:
                res[key].append(value)
    return pd.DataFrame(res)

def decay(array,half_life):
    if half_life  == 0:
        return array[-1]
    L = len(array)
    hl = half_life
    weight = [2**((i-L-1)/hl) for i in range(1,L+1)]
    weight = np.array(weight)/np.sum(weight)

    return np.nansum(weight*array)

def truncate(window):
    median_v = np.nanmean(window)
    distans = [abs(item-median_v) for item in window]
    median_std = np.nanstd(distans)
    res = []
    for term in window:
        if term <= (median_v - 4*median_std):
            res.append(median_v - 4*median_std)
        elif term >= median_v + 4*median_std:
            res.append(median_v + 4*median_std)
        else:
            res.append(term)
    return res



# stk_info = pd.read_excel('utils/股票信息词典.xlsx',encoding='gbk',index_col=0)
stk_ret = pd.read_csv("utils/monthly_return.csv",index_col=0,parse_dates = True)
stk_mkt_cap = pd.read_csv("utils/monthly_cap.csv",index_col=0,parse_dates = True)

def generate_factor_tool(stock_pool=600, partLen_thd=10,side='both',types=['specific','presence'], factor_type='partNum'):
    res = []
    if side == 'buy':
        side_filter = buyside_filter
    elif side == 'sell':
        side_filter = sellside_filter
    else:
        side_filter = bothside_filter

    type_filter = []
    if 'specific' in types:
        type_filter.extend(specific)
    if 'presence' in types:
        type_filter.extend(presence)
    if 'others' in types:
        type_filter.extend(others)
    
    # stock_pool = 600
    if factor_type == 'partNum':
        for doc in collection.find({"activityType":{"$in":type_filter}}):
            partOrgs = doc["partOrgs"] # [{}{}]
            partOrgs_filterd = []
            for part in partOrgs:
                if part["orgType"] in side_filter:
                    partOrgs_filterd.append(part)
            tmp = min(partLen_thd, len(partOrgs_filterd))
            res.append({'_id':{'activityDate':doc["activityDate"],'stkCode':doc["stkCode"]}, 'count':tmp})
   
    elif factor_type == 'OrgNum':
        for doc in collection.find({"activityType":{"$in":type_filter}}):
            partOrgs = doc["partOrgs"] # [{}{}]
            partOrgs_filterd = []
            for part in partOrgs:
                if part["orgType"] in side_filter:
                    partOrgs_filterd.append(part["orgName"])
            tmp = min(5, len(pd.unique(partOrgs_filterd)))
            res.append({'_id':{'activityDate':doc["activityDate"],'stkCode':doc["stkCode"]}, 'count':tmp})
    elif factor_type == 'QuestionNum':
        for doc in collection.find({"activityType":{"$in":type_filter}}):
            tmp = min(15, doc["statementNum"])        
            res.append({'_id':{'activityDate':doc["activityDate"],'stkCode':doc["stkCode"]}, 'count':tmp})
    elif factor_type == 'Dummy':
        for doc in collection.find({"activityType":{"$in":type_filter}}):
            tmp = 1       
            res.append({'_id':{'activityDate':doc["activityDate"],'stkCode':doc["stkCode"]}, 'count':tmp})



    df = transfer2df(res)
    df = df.pivot_table(index='activityDate',columns='stkCode',aggfunc='sum')
    df.columns = df.columns.droplevel(0)
    df_raw = df.reindex(columns = df.apply(np.nansum).sort_values(ascending=False).index)
    df_raw = df_raw.reindex(columns=df_raw.sum().sort_values(ascending=False).index)
    df = df_raw.iloc[:,:stock_pool]
    # df = df_raw
    stk_info = pd.read_excel('utils/股票信息词典.xlsx').astype('<U6')
    stk_info['trade_code'] = stk_info['trade_code'].apply(lambda x: '0'*(6-len(x))+x)
    stk_info = stk_info.set_index("trade_code",drop=False)
    df.columns = [stk_info.loc[i,'wind_code'] for i in df.columns]
    df.to_csv('utils/'+factor_type+'factor_daily_raw.csv')



####分层回测函数
def backtest_tool(df,stk_info,stk_mkt_cap,std_index_ret,port_num=5,factor_type='partNum' ):
    industry_port_panel = {}
    index_ret = {}
    port_dataframe = {} 
    positions_info = []
    dates = df.index
    port_column = ['Port'+str(i+1) for i in range(port_num)]
    num_eachPort = min(150, int(df.shape[1]/port_num))


    ## 准备画每天的仓位bar
    # gridlist = []
    for date_idx in range(len(dates)-1):
        date = dates[date_idx]
        tomorrow = dates[date_idx+1]
        industry_port_day = pd.DataFrame(None,index=industry_list,columns=port_column)
        ## 按照组合数量记录每天的仓位
        positions = {}
        for i in range(port_num):
            positions['Port'+str(i+1)] = defaultdict(list)
        ## !!!!此处有改动
        trading_stks = df.columns[~pd.isnull(df.loc[date])]
        for industry in industry_list:
            stks_industry = trading_stks.intersection(stk_info.index[stk_info['industry_sw'] == industry])
            if len(stks_industry) == 0:
                for i in range(port_num):
                    industry_port_day.loc[industry,'Port'+str(i+1)] = 0
                continue

            stk_scores = df.loc[date,stks_industry]
            stk_scores.sort_values(ascending=False, inplace=True)
            msk = pd.isnull(stk_scores)
            stk_scores = stk_scores[~msk]
            portfolios = my_partition(len(stk_scores), port_num)
            for i,port in enumerate(portfolios):
                stks_pick = stk_scores.index[port['stk_idx']]
                stks_weight = port['weight']
                industry_port_day.loc[industry,'Port'+str(i+1)] = np.sum(stks_weight.values*stk_ret.loc[tomorrow,stks_pick].values)
                # 每个组合选股票
                positions['Port'+str(i+1)]['stks'].extend(stks_pick[stks_weight!=0])
                positions['Port'+str(i+1)]['factor'].extend(df.loc[date,stks_pick[stks_weight!=0]])
                  
        mkt_cap = stk_mkt_cap.loc[date]
        industry_port_panel[tomorrow]=industry_port_day
        index_ret[tomorrow] = np.nansum(mkt_cap/np.nansum(mkt_cap) * stk_ret.loc[tomorrow])
    
        
        industry_weight_list = [stk_info.loc[code,'industry_sw'] for code in mkt_cap.index]
        industry_cap = pd.DataFrame({'mkt_cap':mkt_cap,'industry':industry_weight_list})
        industry_cap_group = industry_cap.groupby('industry').apply(np.nansum)
        industry_weight = industry_cap_group / np.nansum(industry_cap_group)
        industry_weight = industry_weight.reindex(industry_port_day.index)
        port_dataframe[tomorrow] = pd.Series(np.nansum(industry_port_day.values*industry_weight.values.reshape((-1,1)),axis=0), index=port_column)
    
        ## 作画当天的仓位图
        positions_day = {}
        positions_day['day'] = date.strftime('%Y%m%d')
        positions_day['pos'] = []
        for i in range(port_num):
            port_df = pd.DataFrame(positions['Port'+str(i+1)])
            port_df = port_df.sort_values(by='factor',ascending=False).iloc[:num_eachPort,:]
            positions_day['pos'].append({'sec_name':stk_info.loc[port_df['stks'],'sec_name'].tolist(), 'factor':port_df['factor'].tolist(), 'next_ret':stk_ret.loc[tomorrow,port_df['stks']].tolist()})
        positions_info.append(positions_day)

    index_return = pd.Series(index_ret)*0.01
    portfolio_return_df = pd.DataFrame(port_dataframe).T*0.01

    portfolio_return_df['baseline'] = np.mean(portfolio_return_df.values,axis=1)
    portfolio_return_df['index'] = index_return
    portfolio_return_df['hs300'] = std_index_ret.loc[portfolio_return_df.index,'000300.SH']
    portfolio_return_df['zz500'] = std_index_ret.loc[portfolio_return_df.index,'000905.SH']
    portfolio_return_df['zz1000'] = std_index_ret.loc[portfolio_return_df.index,'000852.SH']
    portfolio_return_df['szzz'] = std_index_ret.loc[portfolio_return_df.index,'399106.SZ']
    portfolio_return_df = pd.concat((pd.DataFrame([[0]*len(portfolio_return_df.columns)],columns=portfolio_return_df.columns,index=[dates[0]]),portfolio_return_df))
    portfolio_netvalue_df = (portfolio_return_df+1).apply(np.cumprod)
    portfolio_return_df.to_csv('utils/'+factor_type+'portfolio_return.csv')
    portfolio_netvalue_df.to_csv('utils/'+factor_type+'portfolio_netvalue.csv')


    summary = {}
    benchmark = 'zz500'
    for i in port_column:
        report = analysis(portfolio_return_df[i], portfolio_return_df[benchmark])
        summary[i] = pd.Series(report)
    summary['szzz'] = pd.Series( analysis(portfolio_return_df['szzz'], portfolio_return_df['szzz']) )
    summary['index'] = pd.Series( analysis(portfolio_return_df['index'], portfolio_return_df['zz1000']) )
    summary['zz1000'] = pd.Series( analysis(portfolio_return_df['zz1000'], portfolio_return_df['zz1000']) )
    summary['zz500'] = pd.Series( analysis(portfolio_return_df['zz500'], portfolio_return_df['zz500']) )
    summary['hs300'] = pd.Series( analysis(portfolio_return_df['hs300'], portfolio_return_df['hs300']) )
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv('utils/'+factor_type+'strategy_summary.csv',encoding='gbk')
    draw_down,start_idx,end_idx = analysis(portfolio_return_df['Port1'], portfolio_return_df[benchmark], return_dd=True)


    return positions_info,draw_down.tolist(),start_idx-1,end_idx-1
 
def cal_turnover(previous, today):
    today = set(today)
    previous = set(previous)
    buy = today.difference(previous)
    sell = previous.difference(today)
    return len(buy)/len(previous)

#参数weight：默认为1，对于一篇ariticle的不同部分给予不同的权重
def get_score(word_list):

    # pos_dict = {'times':0,'score':0}
    # neg_dict = {'times':0,'score':0}
    score = 0
    positive = positive_df.index.tolist()
    negative = negative_df.index.tolist()
    positive_index = []
    negative_index = []
    revert_index = []

    for (index,word) in enumerate(word_list):
        word_score = 0
        #判断极性
        if word in positive:
            word_score+=positive_df.loc[word,'score']
            positive_index.append(index)
            '''
            两种情况：
            1. 非常 不 好吃
            2. 不是 很 好吃
            需要极性反转
            '''
            if index-1>=0 and word_list[index-1] in neg_degree: 
                word_score = word_score*(-1)
                revert_index.append(index-1)
            elif index-2>=0 and word_list[index-2] in neg_degree:
                word_score = word_score*(-1)
                revert_index.append(index-2)

        elif word in negative:
            word_score+=negative_df.loc[word,'score']
            negative_index.append(index)
            '''
            1. 不是 不好
            2. 不是 很 不好
            极性反转
            '''
            if index-1>=0 and word_list[index-1] in neg_degree: 
                word_score = word_score*(-1)
                revert_index.append(index-1)
            elif index-2>=0 and word_list[index-2] in neg_degree:
                word_score = word_score*(-1)
                revert_index.append(index-2)

        #判断程度词
        if index-1>=0:
            #赫夫曼二叉树，加权路径最小
            if word_list[index-1] in more_degree or (index-2>=0 and word_list[index-2] in more_degree):
                    word_score = word_score*2
            elif word_list[index-1] in ish_degree or (index-2>=0 and word_list[index-2] in more_degree):
                    word_score = word_score*1.5
            elif word_list[index-1] in very_degree or (index-2>=0 and word_list[index-2] in more_degree):
                    word_score = word_score*2.5
            elif word_list[index-1] in least_degree or (index-2>=0 and word_list[index-2] in more_degree):
                    word_score = word_score*1.1
            elif word_list[index-1] in most_degree or (index-2>=0 and word_list[index-2] in more_degree):
                    word_score = word_score*3
        score += word_score
        # if word_score>0:
        #     pos_dict['times']+=1
        #     pos_dict['score']+=word_score
        # elif word_score<0:
        #     neg_dict['times'] += 1
        #     neg_dict['score'] += word_score
    # print(score,positive_index,negative_index,revert_index)
    return score,positive_index,negative_index,revert_index

def sent_analyse(raw):
    #split sentences
    sentences,num = split_sentences(raw)
    sent_scores = []
    word_cloud = []
    html = str(len(sentences))+'<br>'
    for sentence in sentences:
        words_cut = jieba.lcut(sentence)
        words_list_idx = []
        word_list = []
        for idx,word in enumerate(words_cut):
            if word not in stopwords:
                word_list.append(word)
                words_list_idx.append(idx)
        word_cloud.extend(word_list)
        score,positive_index,negative_index,revert_index = get_score(word_list)

        for idx in positive_index:
            words_cut[words_list_idx[idx]] = '<text class="text-success">'+words_cut[words_list_idx[idx]]+'</text>'
        for idx in negative_index:
            words_cut[words_list_idx[idx]] = '<text class="text-danger">'+words_cut[words_list_idx[idx]]+'</text>'
        for idx in revert_index:
            words_cut[words_list_idx[idx]] = '<text class="text-primary">'+words_cut[words_list_idx[idx]]+'</text>'  
        html_sentence = '/'.join(words_cut)
        html+=html_sentence+'  %.1f<br><br>' % score
        sent_scores.append(score)
        # print(score)
    # print(sent_scores)
    html+='total: %.1f mean: %.1f' % (np.sum(sent_scores), np.mean(sent_scores))   

    return word_cloud, html

def analysis(ret_series_old, benchmark_old, return_dd=False):

    msk = ret_series_old.isnull()
    date_notnull = ret_series_old.index[~msk]
    ret_series = ret_series_old[~msk]
    benchmark = benchmark_old.loc[date_notnull]
    tmp_ret_series = pd.concat((pd.Series([0],index=[ret_series.index[0] - datetime.timedelta(days=-30)]), ret_series))
    tmp_benchmark = pd.concat((pd.Series([0],index=[benchmark.index[0] - datetime.timedelta(days=-30)]), benchmark))

    trading_days = ret_series.index
    asset = np.cumprod(tmp_ret_series + 1)
    L = ret_series.shape[0]
    strat_ret = (asset[-1] / asset[0]) ** (12 / L) - 1
    strat_vol = np.nanstd(ret_series) * np.sqrt(12)

    tmp = asset.cummax()
    draw_down = (tmp - asset)/tmp
    idx = np.argmax(draw_down.values)
    start_idx = asset.tolist().index(tmp[idx])

    performance_string = 'anual ret: %.2f\nanual vol: %.2f\nanual sharp: %.2f\nmax draw down: %.2f\n' \
                         % (strat_ret, strat_vol, strat_ret / strat_vol, draw_down[idx])

    asset_excess = np.cumprod(tmp_ret_series - tmp_benchmark + 1)
    L = ret_series.shape[0]
    excess_ret = (asset_excess[-1] / asset_excess[0]) ** (12 / L) - 1
    excess_vol = np.nanstd(ret_series - benchmark) * np.sqrt(12)

    tmp = asset_excess.cummax()
    excess_draw_down = tmp - asset_excess
    excess_idx = np.argmax(excess_draw_down.values)
    start_excess_idx = asset_excess.tolist().index(tmp[excess_idx])
    report = {}

    report['年化收益率'] = strat_ret
    report['年化波动率'] = strat_vol
    report['夏普比率'] = strat_ret / strat_vol
    report['最大回撤'] = draw_down[idx]
    report['超额收益年化收益率'] = excess_ret
    report['超额收益年化波动率'] = excess_vol
    report['信息比率'] = excess_ret / excess_vol

    report['超额收益胜率'] = np.sum(ret_series - benchmark >= 0) / L

    report['超额收益最大回撤'] = excess_draw_down[excess_idx] / asset_excess[excess_idx]

    if not return_dd:

        return report

    else:

        return (draw_down[1:].apply(lambda x: 0 if x<0 else x), start_idx, idx)


def get_netvalue(ret_df):
    tmp = pd.concat((pd.DataFrame([[0] * len(ret_df.columns)], columns=ret_df.columns,
                                  index=[ret_df.index[0] - datetime.timedelta(days=-30)]), ret_df))
    return (tmp+1).apply(np.cumprod)

def get_summary(ret_df, benchmark):
    summary = {}
    for col in ret_df.columns:
        report = analysis(ret_df[col], ret_df[benchmark])
        summary[col] = pd.Series(report)
    summary_df = pd.DataFrame(summary).T
    # summary_df = summary_df.applymap(lambda x:np.round(x,3))
    columns = ['组合']+summary_df.columns.tolist()
    summary_df['组合'] = summary_df.index.tolist()
    summary_df = summary_df.reindex(columns=columns)
    summary_df = summary_df.fillna(value=0)
    return summary_df

class NameForm(FlaskForm):
    stkCode = StringField("查询股票或行业", validators=[DataRequired()])
    start_date = StringField("开始日期")
    end_date = StringField("截止日期")
    submit = SubmitField('查询')

class SentForm(FlaskForm):
    sent_raw = StringField("请输入文字")
    submit = SubmitField('提交')


def factor_cov_matrix():
    factors = ['partNum', 'OrgNum', 'QuestionNum', 'Dummy']
    factor0 = pd.read_csv('utils/' + factors[0] + 'backtest_factor.csv', index_col=0, parse_dates=True)
    stocks = factor0.columns
    factor_dfs = []
    for factor in factors:
        factor_df = pd.read_csv('utils/' + factor + 'backtest_factor.csv', index_col=0, parse_dates=True)
        stocks = stocks.intersection(factor_df.columns)
        factor_dfs.append(factor_df)

    res = []
    for date in factor_df.index:
        tmp = pd.DataFrame({factors[i]: factor_dfs[i].loc[date, stocks] for i in range(4)}, columns=factors)
        res.append(tmp.corr().values.tolist())
    res = np.mean(np.array(res), axis=0)