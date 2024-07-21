# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 02:07:53 2024

@author: 385225
"""


import os
os.chdir('C:\\Users\\385225\\Documents\\GitHub\\Barra')



from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



    
#    計算Newey-West調整過的協方差矩陣
#    ret: DataFrame, 行為時間，列為因子收益
#    q: 假設因子收益為q階MA過程
#    tao: 算協方差時的半衰期
#    h: 樣本時間長度
#    D: 協方差計算時的最大滯後時間長度


def Newey_West(ret, q = 2, tao = 90, h = 252):
    '''
    Newey_West方差調整
    時序上存在相關性時，使用Newey_West調整協方差估計
    factor_ret: DataFrame, 行為時間，列為因子收益
    q: 假設因子收益為q階MA過程
    tao: 算協方差時的半衰期
    '''
    from functools import reduce
    from statsmodels.stats.weightstats import DescrStatsW 
    
    T = ret.shape[0]           #时序长度
    K = ret.shape[1]           #因子数
    if T <= q :
        raise Exception("T <= q ")
    elif  T <= K :
        raise Exception(" T <= K ")
         
    names = ret.columns    
    weights = 0.5**(np.arange(T-1,-1,-1)/tao)   #指数衰减权重
    weights = weights / sum(weights)
    
    w_stats = DescrStatsW(ret, weights)
    ret = ret - w_stats.mean
    
    ret = np.matrix(ret.values)
    Gamma0 = [weights[t] * ret[t].T  @ ret[t] for t in range(max(T - h, 0), T)]
    Gamma0 = reduce(np.add, Gamma0)
    
    
    V = Gamma0             #调整后的协方差矩阵
    for i in range(1,q+1):
        Gammai = [weights[i+t] * ret[t].T  @ ret[i+t] for t in range(max(T - h, 0), T - i)]
        Gammai = reduce(np.add, Gammai)
        V += (1 - i/(1+q)) * (Gammai + Gammai.T)
    V * 22
    return(pd.DataFrame(V, columns = names, index = names))

"""
def Newey_West(ret, q = 2, tao = 252):
        '''
        Newey_West方差調整
        時序上存在相關性時，使用Newey_West調整協方差估計
        factor_ret: DataFrame, 行為時間，列為因子收益
        q: 假設因子收益為q階MA過程
        tao: 算協方差時的半衰期
        '''
        from functools import reduce
        from statsmodels.stats.weightstats import DescrStatsW 
        
        T = ret.shape[0]           #时序长度
        K = ret.shape[1]           #因子数
        if T <= q or T <= K:
            raise Exception("T <= q or T <= K")
             
        names = ret.columns    
        weights = 0.5**(np.arange(T-1,-1,-1)/tao)   #指数衰减权重
        weights = weights / sum(weights)
        
        w_stats = DescrStatsW(ret, weights)
        ret = ret - w_stats.mean
        
        ret = np.matrix(ret.values)
        Gamma0 = [weights[t] * ret[t].T  @ ret[t] for t in range(T)]
        Gamma0 = reduce(np.add, Gamma0)
        
        
        V = Gamma0             #调整后的协方差矩阵
        for i in range(1,q+1):
            Gammai = [weights[i+t] * ret[t].T  @ ret[i+t] for t in range(T-i)]
            Gammai = reduce(np.add, Gammai)
            V = V + (1 - i/(1+q)) * (Gammai + Gammai.T)
        
        return(pd.DataFrame(V, columns = names, index = names))


"""

def eigen_risk_adj(covmat, T = 1000, M = 100, scale_coef = 1.4):
    '''
    Eigenfactor Risk Adjustment
    T: 序列長度
    M: 模擬次數
    scale_coef: 偏差的縮放係數
    '''
    F0 = covmat
    K = covmat.shape[0]
    D0,U0 = np.linalg.eig(F0)      #特征值分解; D0是特征因子组合的方差; U0是特征因子组合中各因子权重; F0是因子协方差方差
    #F0 = U0 @ D0 @ U0.T    D0 = U0.T @ F0 @ U0  
    
    if not all(D0>=0):         #检验正定性
        raise('covariance is not symmetric positive-semidefinite')
   
    v = []  #bias
    for m in range(M):
        ## 模拟因子协方差矩阵
        np.random.seed(m+1)
        bm = np.random.multivariate_normal(mean = K*[0], cov = np.diag(D0), size = T).T  #特征因子组合的收益
        fm = U0 @ bm       #反变换得到各个因子的收益
        Fm = np.cov(fm)    #模拟得到的因子协方差矩阵

        ##对模拟的因子协方差矩阵进行特征分解
        Dm,Um = np.linalg.eig(Fm)   # Um.T @ Fm @ Um 
    
        ##替换Fm为F0
        Dm_hat = Um.T @ F0 @ Um 

        v.append(np.diagonal(Dm_hat) / Dm)

    v = np.sqrt(np.mean(np.array(v), axis = 0))
    v = scale_coef * (v-1) + 1
    
    
    D0_hat = np.diag(v**2) * np.diag(D0)  #调整对角线
    F0_hat = U0 @ D0_hat @ U0.T           #调整后的因子协方差矩阵
    return(pd.DataFrame(F0_hat, columns = covmat.columns, index = covmat.columns))

    


def eigenfactor_bias_stat(cov, ret, predlen = 1):
    '''
    計算特徵因子組合的偏差統計量
    '''
    #bias stat
    b = []
    for i in range(len(cov)-predlen):
        try:
            D, U = np.linalg.eig(cov[i])                              #特征分解, U的每一列就是特征因子组合的权重
            U = U / np.sum(U, axis = 0)                               #将权重标准化到1
            sigma = np.sqrt(predlen * np.diag(U.T @ cov[i] @ U))      #特征因子组合的波动率
            retlen = (ret.values[(i+1):(i+predlen+1)] + 1).prod(axis=0) - 1
            r = U.T @ retlen                                          #特征因子组合的收益率
            b.append(r / sigma)
        except:
            pass
    
    b = np.array(b)
    bias_stat = np.std(b, axis = 0)
    plt.plot(bias_stat)
    return(bias_stat)
    


 
    
def progressbar(cur, total, txt):
    '''
    显示进度条
    '''
    percent = '{:.2%}'.format(cur / total)
    print("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent) + txt, end = '')
    



def group_mean_std(x):
    '''
    计算一组的加权平均和波动率
    '''
    m =sum(x.volatility*x.capital) / sum(x.capital)
    s = np.sqrt(np.mean((x.volatility - m)**2))
    return([m, s])


def shrink(x, group_weight_mean, q):
    '''
    计算shrink估计量
    '''
    a = q * np.abs(x['volatility'] - group_weight_mean[x['group']][0])
    b =  group_weight_mean[x['group']][1]
    v = a / (a + b)     #收缩强度
    SH_est = v * group_weight_mean[x['group']][0] + (1-v) * np.abs(x['volatility'])    #贝叶斯收缩估计量
    return(SH_est)
    

def bayes_shrink(volatility, capital, ngroup = 10, q = 1):
    '''
    使用市值對特異性收益率波動率進行貝葉斯收縮，以保證波動率估計在樣本外的持續性
    volatility: 波動率
    capital: 市值
    ngroup: 劃分的組數
    q: 收縮參數
    '''
    group = pd.qcut(capital, ngroup).codes    #按照市值分为10组
    data = pd.DataFrame(np.array([volatility, capital, group]).T, columns = ['volatility', 'capital', 'group'])
    #分组计算加权平均
    grouped = data.groupby('group')
    group_weight_mean = grouped.apply(group_mean_std)
    
    SH_est = data.apply(shrink, axis = 1, args = (group_weight_mean, q))   #贝叶斯收缩估计量 
    return(SH_est.values)


def style_factor_norm(factors, capital):
    '''
    使用市值进行标准化
    '''
    from statsmodels.stats.weightstats import DescrStatsW
    w_stats = DescrStatsW(factors, weights = capital)
    w_mu = w_stats.mean                   # 加权平均
    w_std = np.std(factors)        # 等权标准差
    return((factors - w_mu) / w_std)




class CrossSection():
    def __init__(self, data, robust_ret=None, jump = None, log_forecast = None ):
        '''
        if data.empty:
            raise ValueError(f"輸入的數據為空，日期: {data['date'].iloc[0] if 'date' in data.columns else 'unknown'}")
        if 'date' not in data.columns:
            raise ValueError("數據中缺少 'date' 列")
        if 'stocknames' not in data.columns:
            raise ValueError("數據中缺少 'stocknames' 列")
        '''
        #print(f"CrossSection 初始化，數據行數: {len(data)}, 列: {data.columns.tolist()}")
        #print("Initializing CrossSection...")
        #print("Data Columns:", data.columns)
        
        # 拆分数据
        self.date = data['date'].values[0]
        self.stocknames = data['stocknames'].values
        self.capital = data['capital'].values
        self.ret = data['ret'].values
        self.robust_ret =  robust_ret
        self.specific_ret = None
        self.log_forecast = log_forecast
        self.jump = jump
        
        style_factor_columns = ['Size', 'Liquidity', 'Momentum', 'Volatility', 'Value', 'Earning', 'Dividend', 'Leverage', 'Growth']
        industry_factor_columns = ['M1100', 'M1200', 'M1300', 'M1400', 'M1500', 'M1600', 'M1721', 'M1722', 'M1800', 'M1900', 'M2000', 'M2100', 'M2200',
                                   'M2324', 'M2325', 'M2326', 'M2327', 'M2328', 'M2329', 'M2330', 'M2331', 'M2500', 'M2600', 'M2700', 'M2900', 'M3500',
                                   'M3600', 'M3700', 'M3800', 'M9700', 'M9900']
        
        self.style_factors = data[style_factor_columns].values
        self.industry_factors = data[industry_factor_columns].values
        
        self.N = data.shape[0]
        self.Q = len(style_factor_columns)
        self.P = len(industry_factor_columns)
        
        self.style_factors_names = style_factor_columns
        self.industry_factors_names = industry_factor_columns
        self.country_factors = np.array(self.N * [[1]])
        self.cap_w = (self.capital) / sum((self.capital))
        self.W = np.sqrt(self.capital) / sum(np.sqrt(self.capital))
        self.cap_w95 = np.clip(np.sqrt(self.capital) / sum(np.sqrt(self.capital)), None, np.quantile(np.sqrt(self.capital) / sum(np.sqrt(self.capital)), 0.95))
        #print(self.W95)
        #print(f'Cross Section Regression, Date: {self.date}, {self.N} Stocks, {self.P} Industry Factors, {self.Q} Style Factors')

    
    def reg1(self):
        '''
        多因子模型求解
        '''
        
        #W = np.diag(self.W)
        W = np.diag(self.cap_w)
        
        if self.P>0:
            #各个行业的总市值
            industry_capital = np.array([sum(self.industry_factors[:,i] * self.capital) for i in range(self.P)])
            

            #處理行業共線性而引入的中性限制對應變換矩陣R
            R = np.eye(1 + self.P + self.Q)    
            R[self.P, 1:(1+self.P)] = -industry_capital / industry_capital[-1]
            R = np.delete(R, self.P, axis =1)
            
            #求解多因子模型
            factors = np.matrix(np.hstack([self.country_factors, self.industry_factors, self.style_factors]))  
            factors_tran = factors @ R
            pure_factor_portfolio_weight = R @ np.linalg.inv(factors_tran.T @ W @ factors_tran) @ factors_tran.T @ W  #纯因子组合权重
            
            '''
            國家因子：第一行表示國家因子純因子組合在各個因子上的曝露，
                     國家因子純因子組合在國家因子上曝露為1，在風格因子上曝露為0，
                     但是在行業因子上不為0（曝露恰好為行業市值權重）
                     而在行業中性限制下國家因子純因子組合在行業因子曝露上獲得的收益恰好為0
                     從而國家因子純因子組合的收益就是國家因子的收益 f_c
            
            行業因子：第2行-第2+P行表示行業純因子組合在各個因子上的曝露，
                     行業因子純因子組合在國家因子上曝露為0，在風格因子上曝露為0，
                     但是在各個行業因子上不為0，
                     這裡所謂的行業純因子組合是指做多行業純因子，同時做空國家純因子組合，
                     為了得到真正的行業純因子，應該是第一行+第2行，獲得的收益是國家因子收益+該行業的純因子收益，
                     這裡算出來的行業因子純因子組合的收益應該理解為行業因子的收益與國家因子的相對收益
            
            風格因子：在風格因子上曝露為1，其他因子上曝露為0，收益為風格純因子的收益

            '''
            
        else:
            #求解多因子模型
            factors = np.matrix(np.hstack([self.country_factors, self.style_factors]))
            
            pure_factor_portfolio_weight = np.linalg.inv(factors.T @ W @ factors) @ factors.T @ W    #純因子組合權重

        
        factor_ret = pure_factor_portfolio_weight @ self.ret                        #純因子收益
        #print(len(factors))
        factor_ret = np.array(factor_ret)[0]
        #factor_daily_expousure = factors
        pure_factor_portfolio_exposure = pure_factor_portfolio_weight @ factors     #纯因子组合在各个因子上的暴露
        specific_ret = self.ret - np.array(factors @ factor_ret.T)[0]               #個股特異收益率 (Un)
        #R2 = 1 - np.var(specific_ret) / np.var(self.ret)                            #R square
        
        #return((factor_ret, specific_ret, pure_factor_portfolio_exposure, R2))
        return((factor_ret, specific_ret, pure_factor_portfolio_exposure))

    def reg2(self):
        '''
        多因子模型求解
        '''
        
        #W = np.diag(self.W)
        W = np.diag(self.cap_w)
        
    
        if self.P>0:
            #各个行业的总市值
            industry_capital = np.array([sum(self.industry_factors[:,i] * self.capital) for i in range(self.P)])
            

            #處理行業共線性而引入的中性限制對應變換矩陣R
            R = np.eye(1 + self.P + self.Q)    
            R[self.P, 1:(1+self.P)] = -industry_capital / industry_capital[-1]
            R = np.delete(R, self.P, axis =1)
            
            #求解多因子模型
            factors = np.matrix(np.hstack([self.country_factors, self.industry_factors, self.style_factors]))  
            factors_tran = factors @ R
            pure_factor_portfolio_weight = R @ np.linalg.inv(factors_tran.T @ W @ factors_tran) @ factors_tran.T @ W  #纯因子组合权重
            
            '''
            國家因子：第一行表示國家因子純因子組合在各個因子上的曝露，
                     國家因子純因子組合在國家因子上曝露為1，在風格因子上曝露為0，
                     但是在行業因子上不為0（曝露恰好為行業市值權重）
                     而在行業中性限制下國家因子純因子組合在行業因子曝露上獲得的收益恰好為0
                     從而國家因子純因子組合的收益就是國家因子的收益 f_c
            
            行業因子：第2行-第2+P行表示行業純因子組合在各個因子上的曝露，
                     行業因子純因子組合在國家因子上曝露為0，在風格因子上曝露為0，
                     但是在各個行業因子上不為0，
                     這裡所謂的行業純因子組合是指做多行業純因子，同時做空國家純因子組合，
                     為了得到真正的行業純因子，應該是第一行+第2行，獲得的收益是國家因子收益+該行業的純因子收益，
                     這裡算出來的行業因子純因子組合的收益應該理解為行業因子的收益與國家因子的相對收益
            
            風格因子：在風格因子上曝露為1，其他因子上曝露為0，收益為風格純因子的收益

            '''
            
        else:
            #求解多因子模型
            factors = np.matrix(np.hstack([self.country_factors, self.style_factors]))
            #factors = np.matrix(np.hstack([self.country_factors, self.industry_factors, self.style_factors]))
            pure_factor_portfolio_weight = np.linalg.inv(factors.T @ W @ factors) @ factors.T @ W    #純因子組合權重
        
        
        factor_ret = pure_factor_portfolio_weight @ self.robust_ret                        #純因子收益
        
        #print((factors.shape))
        factor_ret = np.array(factor_ret)[0]
        pure_factor_portfolio_weight = np.array(pure_factor_portfolio_weight)[0]
        
        #print('ok')
        pure_factor_portfolio_exposure = pure_factor_portfolio_weight @ factors     #纯因子组合在各个因子上的暴露
        #pure_factor_portfolio_exposure = np.array(pure_factor_portfolio_weight)[0]
        #print(pure_factor_portfolio_exposure)
        part_specific_ret = self.ret - np.array(factors @ factor_ret.T)[0]
        # self.jump + 
        specific_ret = part_specific_ret
        factor_exposure = factors.T @ self.cap_w
        #print(self.jump)
        #print('\n ---------------------------------------------------------------------')
        #print(part_specific_ret)
        #print('\n =====================================================================')
        #specific_ret = self.specific_ret
        #specific_ret = self.ret - np.array(factors @ factor_ret.T)[0]               #個股特異收益率 (Un)
        #specific_ret = specific_ret.tolist()
        R2 = 1 - (self.cap_w * specific_ret * specific_ret).sum() / (self.cap_w * self.ret *  self.ret).sum()                          #R square
        #R2 = 1 - np.var(specific_ret) / np.var(self.ret)
            
        #specific_ret = specific_ret.tolist()
        #print('alright')
        #return((factor_ret, specific_ret, pure_factor_portfolio_exposure, R2))
        return((factor_ret,  pure_factor_portfolio_exposure, R2, specific_ret, factor_exposure, self.cap_w))
    def reg3(self):
        '''
        特意風險模型求解
        傳入log_forecast
        '''
        
        #W = np.diag(self.W)
        W = np.diag(self.cap_w95)
        
    
        
        factors = np.matrix(np.hstack([self.country_factors, self.industry_factors, self.style_factors]))  
            #factors = np.matrix(np.hstack([self.country_factors, self.industry_factors, self.style_factors]))
        pure_factor_portfolio_weight = np.linalg.inv(factors.T @ W @ factors) @ factors.T @ W    #純因子組合權重
        
        
        log_forecast = pure_factor_portfolio_weight @ self.log_forecast                        #純因子收益
        
        log_forecast = np.array(log_forecast)[0]
        #print(type(factor_ret))
        #print('ok')
        pure_factor_portfolio_exposure = pure_factor_portfolio_weight @ factors     #纯因子组合在各个因子上的暴露 #expect return
        
    
        return((log_forecast,  pure_factor_portfolio_exposure))
    
#        

'''
class MFM():

    def __init__(self, data, P, Q):
        self.Q = Q                                                           #风格因子数
        self.P = P                                                           #行业因子数
        self.dates = pd.to_datetime(data.date.values)                        #日期
        self.sorted_dates = pd.to_datetime(np.sort(pd.unique(self.dates)))   #排序后的日期
        self.T = len(self.sorted_dates)                                      #期数
        self.data = data                                                     #数据
        self.columns = ['country']                                           #因子名
        self.columns.extend((list(data.columns[4:])))
        
        self.last_capital = None                                             #最后一期的市值 
        self.factor_ret = None                                               #因子收益
        self.specific_ret = None                                             #特异性收益
        self.R2 = None                                                       #R2
        
        self.Newey_West_cov = None                        #逐时间点进行Newey West调整后的因子协方差矩阵
        self.eigen_risk_adj_cov = None                    #逐时间点进行Eigenfactor Risk调整后的因子协方差矩阵
        self.vol_regime_adj_cov = None                    #逐时间点进行Volatility Regime调整后的因子协方差矩阵
    
'''    

class MFM():
    def __init__(self, data, P, Q):
        self.Q = Q                                                           #风格因子数
        self.P = P                                                           #行业因子数
        self.dates = pd.to_datetime(data.date.values)                        #日期
        self.sorted_dates = pd.to_datetime(np.sort(pd.unique(self.dates)))   #排序后的日期
        #self.ret = data['ret']
        self.T = len(self.sorted_dates)                                      #期数
        self.data = data                                                     #数据
        self.columns = ['country']                                           #因子名
        self.columns.extend(['Size', 'Liquidity', 'Momentum', 'Volatility', 'Value', 'Earning', 'Dividend', 'Leverage', 'Growth'])
        self.columns.extend(['M1100', 'M1200', 'M1300', 'M1400', 'M1500', 'M1600', 'M1721', 'M1722', 'M1800', 'M1900', 'M2000', 'M2100', 'M2200',
                             'M2324', 'M2325', 'M2326', 'M2327', 'M2328', 'M2329', 'M2330', 'M2331', 'M2500', 'M2600', 'M2700', 'M2900', 'M3500',
                             'M3600', 'M3700', 'M3800', 'M9700', 'M9900'])
        #self.columns.extend(['excess_ret'])
        
        self.last_capital = None                                             #最后一期的市值 
        self.factor_ret = None                                               #因子收益
        self.ret = self.data['ret'].values
        self.robust_ret = None                                               #穩定ret
        self.jump = None                                                     #jump
        self.specific_ret = None                                             #特异性收益
        self.Un = None                                                       # specific_ret的df
        self.R2 = None                                                       #R2
        
        self.Newey_West_cov = None                        #逐时间点进行Newey West调整后的因子协方差矩阵
        self.eigen_risk_adj_cov = None                    #逐时间点进行Eigenfactor Risk调整后的因子协方差矩阵
        self.vol_regime_adj_cov = None                    #逐时间点进行Volatility Regime调整后的因子协方差矩阵
        self.specific_risk = None
        self.forecast_risk = None
        self.pure_factor_portfolio_exposure = None
        self.resid = None
        
    def calculate_specific_risk(self, variant='EUE3S'):
        specific_risk_model = SpecificRisk(self, variant)
        self.specific_risk = specific_risk_model.run()
        return self.specific_risk
    
 #   def calculate_specific_risk(self, variant='EUE3S'):
 #       specific_risk_model = SpecificRisk(self, variant)
 #       return specific_risk_model.run()
    
    def reg_by_time(self):
        '''
        逐时间点进行横截面多因子回归
        '''
        factor_ret = []
        R2 = []
        specific_ret = []
        specific_ret_list = [] #********************
        
    
   
        print('===================================逐時第一次横截面多因子回歸===================================')       
        for t in tqdm(range(self.T), desc='第一次横截面多因子回歸'):
            data_by_time = self.data.iloc[self.dates == self.sorted_dates[t],:]
            data_by_time = data_by_time.sort_values(by = 'stocknames')
            
            cs = CrossSection(data_by_time)
            factor_ret_t, specific_ret_t, _  = cs.reg1()
            #print((factor_ret_t.dtpyes))
            factor_ret.append(factor_ret_t)
            #注意：每个截面上股票池可能不同
            specific_ret.append(pd.DataFrame([specific_ret_t], columns = cs.stocknames, index = [self.sorted_dates[t]]))
            specific_ret_list.append(pd.DataFrame(specific_ret_t, index=cs.stocknames, columns=[self.sorted_dates[t]]).T) #*****************
            
            self.last_capital = cs.capital
        factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)
        
        
        #*-*-*-*-=================================================================================================
        Un = pd.concat(specific_ret_list).sort_index()
        self.Un = Un
        self.specific_ret = specific_ret                                           #特异性收益
        return_jump_df = pd.DataFrame(index=self.dates, columns=pd.Index(self.columns).unique())
        
        #robust_std_df
        # robust_std_df: 每支股票自己的 robust std
        def calculate_robust_std(stock_data):
            return 1.4826 * (abs(stock_data - stock_data.median())).median()
    
    # 计算每支股票的 robust std，结果是一个只有一行的 DataFrame
        robust_std_df = pd.DataFrame(Un.apply(calculate_robust_std, axis=0)).T
        #print(robust_std_df)
        #robust_std_df.columns = Un.columns
        ret_wide = self.data.pivot(index='date', columns='stocknames', values='ret')
        ret_wide.index = pd.to_datetime(ret_wide.index)  # 確保日期索引格式一致
        #print(ret_wide)
        
        return_jump_df = pd.DataFrame(index=ret_wide.index, columns=ret_wide.columns)
        jump = []
        print('===================================正在計算Jump===================================')
        for t in tqdm(range(self.T), desc='計算Jump'):
            period_residual = self.Un.loc[self.sorted_dates[t]]
            ret = ret_wide.loc[self.sorted_dates[t]]
            
            return_jump = np.where(np.abs(period_residual) > 4 * robust_std_df, 
                                   np.sign(period_residual) * (np.abs(period_residual) - 4 * robust_std_df), 
                                   0)
            #print(type(ret))
            #print(type(return_jump))
             
            return_jump_df.loc[self.sorted_dates[t], Un.columns] = ret.values - return_jump
            return_jump = return_jump.flatten()
            jump.append(pd.DataFrame([return_jump], columns = cs.stocknames, index = [self.sorted_dates[t]])) 
            #jump.append(pd.DataFrame([return_jump], columns = cs.stocknames, index = [self.sorted_dates[t]])) 
        #print("出jump迴圈")
        #print(return_jump_df)
        robust_ret_df = return_jump_df
        
        self.jump = jump
        #print(jump)
        
        # 將 robust_ret_df 轉換成長資料格式
        robust_ret_long = robust_ret_df.reset_index().melt(id_vars=['date'], var_name='stocknames', value_name='robust_ret')
        robust_ret_long['date'] = robust_ret_long['date'].astype(str)
        #print("robust_ret_long 類型:")
        #print(robust_ret_long['date'].dtypes)
        #print("self.data 類型:")
        #print(self.data['date'].dtypes)
        #print(robust_ret_long)
        # 將 robust_ret_long 合併回 self.data
        self.data = self.data.merge(robust_ret_long, on=['date', 'stocknames'], how='left')
        
        self.robust_ret = self.data['robust_ret'].values
        #*-*-*-*-=================================================================================================
                
        
        
        
        print('===================================逐時第二次横截面多因子回歸===================================')  
        factor_ret = []
        
        factors = []
        cap_weight = []
        for t in tqdm(range(self.T), desc='第二次横截面多因子回歸'):
            data_by_time = self.data.iloc[self.dates == self.sorted_dates[t],:]
            data_by_time = data_by_time.sort_values(by = 'stocknames')
            
            jump_df = self.jump[t]  # 从self.jump列表中获取第t个DataFrame
            jump_list = jump_df.values.flatten()  # df to numpy.ndarray
            
            cs = CrossSection(data_by_time, robust_ret=data_by_time['robust_ret'].values, jump = jump_list )
            factor_ret_t, _ , R2_t, specific_ret_t, factors_t, cap_weight_t = cs.reg2()
            
            factor_ret.append(factor_ret_t)
            #注意：每个截面上股票池可能不同
            specific_ret.append(pd.DataFrame([specific_ret_t], columns = cs.stocknames, index = [self.sorted_dates[t]]))
            factors.append(np.asarray(factors_t).flatten())
            # 打印調試信息
            #print(f"Date: {self.sorted_dates[t]}")
            #print(type(factors_t.flatten()))
            #print(f"factor_exposure_t shape: {factors_t.flatten().shape}")
            cap_weight.append(cap_weight_t)
            
            R2.append(R2_t)
            self.last_capital = cs.capital
            
        factors = np.array(factors)
        #print(f"factor_exposure_t shape: {factors.shape}")
        cap_weight = pd.DataFrame(cap_weight, columns = cs.stocknames, index = self.sorted_dates)
        #print(cap_weight)
        pure_factor_portfolio_exposure = pd.DataFrame(factors, columns = self.columns, index = self.sorted_dates)
        #print(pure_factor_portfolio_exposure)
        factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)
        R2 = pd.DataFrame(R2, columns = ['R2'], index = self.sorted_dates)
        #Un = pd.concat(specific_ret_list).sort_index()               #***************************************
        
        print('\n===================================完成===================================')
     #*******************************************************************************************************   
        self.factor_ret = factor_ret                                               #因子收益
        self.factors = pure_factor_portfolio_exposure
        self.cap_weight = cap_weight
        self.Un = Un                                                               #特异性收益
        self.R2 = R2                                                               #R2
        return((factor_ret, specific_ret, R2, Un))


 #   def specific_risk(self):
     

    def Newey_West_by_time2(self, q = 2, tao = 90, h = 252):
        '''
        逐時計算斜方差並且進行Newey West調整
        q: 假设因子收益為q皆MA過程
        tao: 算斜方差時的半衰期
        '''
        
        if self.factor_ret is None:
            raise Exception('please run reg_by_time to get factor returns first')
            
        Newey_West_cov = []
        print('\n\n===================================逐時Newey West調整=================================')    
        for t in range(1,self.T+1):
            try:
                Newey_West_cov.append(Newey_West(self.factor_ret[:t], q, tao, h))
            except:
                Newey_West_cov.append(pd.DataFrame())
           # try:
           #     cov_matrix = Newey_West(self.factor_ret[:t], q, tao, h)
           #     Newey_West_cov[self.sorted_dates[t-1]] = cov_matrix
           # except:
           #     Newey_West_cov[self.sorted_dates[t-1]] = pd.DataFrame()
                
            progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])
        
        self.Newey_West_cov = Newey_West_cov
        return(Newey_West_cov)


    def Newey_West_by_time(self, q=2, tao=90, h=252):
        '''
        逐時計算斜方差並且進行Newey West調整
        q: 假设因子收益為q階MA過程
        tao: 算斜方差時的半衰期
        h: 半衰期常數
        *加一個檢查為空就不要加了
        '''
        
        if self.factor_ret is None:
            raise Exception('please run reg_by_time to get factor returns first')
            
        Newey_West_cov = {}  # 改為字典
        print('\n\n===================================逐時Newey West調整=================================')    
        
        for t in tqdm(range(1, self.T+1), desc='Newey West調整'):
            current_date = self.sorted_dates[t-1]
            try:
                cov_matrix = Newey_West(self.factor_ret[:t], q, tao, h)
                Newey_West_cov[current_date] = cov_matrix
            except :
                #print(f"Error at date {current_date}: {str(e)}")
                Newey_West_cov[current_date] = pd.DataFrame()
            #except Exception as e:
            #    print(f"Error at date {current_date}: {str(e)}")
            # 移除 progressbar，因為我們現在使用 tqdm
        
        self.Newey_West_cov = Newey_West_cov
        return Newey_West_cov
    
    def eigen_risk_adj_by_time(self, M = 100, scale_coef = 1.4):
        '''
        逐時Eigenfactor Risk Adjustment
        M: 模擬次数
        scale_coef: scale coefficient for bias
        '''
        
        if self.Newey_West_cov is None:
            raise Exception('please run Newey_West_by_time to get factor return covariances after Newey West adjustment first')        
        
        eigen_risk_adj_cov = []
        print('\n\n===================================逐時Eigenfactor Risk調整=================================')    
        for t in range(self.T):
            try:
                eigen_risk_adj_cov.append(eigen_risk_adj(self.Newey_West_cov[t], self.T, M, scale_coef))
            except:
                eigen_risk_adj_cov.append(pd.DataFrame())
            
            progressbar(t+1, self.T, '   date: ' + str(self.sorted_dates[t])[:10])
        
        self.eigen_risk_adj_cov = eigen_risk_adj_cov
        return(eigen_risk_adj_cov)
        
        
    
    def vol_regime_adj_by_time(self, tao = 84):
        '''
        Volatility Regime Adjustment
        tao: Volatility Regime Adjustment的半衰期
        '''
        
        if self.eigen_risk_adj_cov is None:
            raise Exception('please run eigen_risk_adj_by_time to get factor return covariances after eigenfactor risk adjustment first')        
        
        
        K = len(self.eigen_risk_adj_cov[-1])
        factor_var = list()
        for t in range(self.T):
            factor_var_i = np.diag(self.eigen_risk_adj_cov[t])
            if len(factor_var_i)==0:
                factor_var_i = np.array(K*[np.nan])
            factor_var.append(factor_var_i)
         
        factor_var = np.array(factor_var)
        B = np.sqrt(np.mean(self.factor_ret**2 / factor_var, axis = 1))      #截面上的bias统计量
        weights = 0.5**(np.arange(self.T-1,-1,-1)/tao)                            #指数衰减权重
        
        
        lamb = []
        vol_regime_adj_cov = []
        print('\n\n==================================逐時Volatility Regime調整================================') 
        for t in range(1, self.T+1):
            #取除无效的行
            okidx = pd.isna(factor_var[:t]).sum(axis = 1) == 0 
            okweights = weights[:t][okidx] / sum(weights[:t][okidx])
            fvm = np.sqrt(sum(okweights * B.values[:t][okidx]**2))   #factor volatility multiplier
            
            lamb.append(fvm)  
            vol_regime_adj_cov.append(self.eigen_risk_adj_cov[t-1] * fvm**2)
            progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])
        
        self.vol_regime_adj_cov = vol_regime_adj_cov
        return((vol_regime_adj_cov, lamb))
    
#==============================================================================================================================

    def calculate_forecast_risk(self):
        if self.resid is None:
            raise ValueError("resid must be calculated before calculating 特定風險")
            
        factor_risk = self.Newey_West_cov
        weights = self.cap_weight.reindex(columns=self.resid.columns)
        weights = weights.fillna(0)
        resid = self.resid
        portfolio_exposure = self.factors
        forecast_risk = pd.Series(np.nan, index=factor_risk.keys(), name='forecast_risk')
        
        for index, date in tqdm(enumerate(forecast_risk.index), desc='計算 forecast risk'):
            try:
                period_weights = weights.loc[date].fillna(0)
                period_portfolio_exposure = portfolio_exposure.loc[date]
                period_factor_risk = factor_risk[date].dropna()
                period_resid = np.diag(resid.loc[date].dropna() ** 2)
                
                if period_factor_risk.empty:
                    #print(f"Warning: Empty factor_risk for date {date}")
                    continue
                
                # 添加維度檢查
              #  print("\n------------------------------------------------------------------------------\n")
              #  print(f"\nDate: {date}")
              #  print(f"period_portfolio_exposure shape: {period_portfolio_exposure.shape}")
              #  print(f"period_factor_risk shape: {period_factor_risk.shape}")
              #  print(f"period_weights shape: {period_weights.shape}")
              #  print(f"period_resid shape: {period_resid.shape}")
                
                # 確保 period_portfolio_exposure 是二維數組
                if period_portfolio_exposure.ndim == 1:
                    period_portfolio_exposure = period_portfolio_exposure.values.reshape(1, -1)
                else:
                    period_portfolio_exposure = period_portfolio_exposure.values
                
                # 確保 period_weights 是二維數組
                period_weights = period_weights.values.reshape(-1, 1)
                
               # print(f"After reshape:")
               # print(f"period_portfolio_exposure shape: {period_portfolio_exposure.shape}")
               # print(f"period_weights shape: {period_weights.shape}")
                
                if period_portfolio_exposure.shape[1] != period_factor_risk.shape[0]:
                    print(f"Dimension mismatch: period_portfolio_exposure columns ({period_portfolio_exposure.shape[1]}) != period_factor_risk rows ({period_factor_risk.shape[0]})")
                    continue
                
                # 分步驟計算並檢查中間結果
                step1 = period_portfolio_exposure @ period_factor_risk
                #print(f"Step 1 shape: {step1.shape}")
                
                step2 = step1 @ period_portfolio_exposure.T
                #print(f"Step 2 shape: {step2.shape}")
                #print(period_weights.shape)
                step3 = period_weights.T @ period_resid
               # print(f"Step 3 shape: {step3.shape}")
                
                step4 = step3 @ period_weights
                #print(f"Step 4 shape: {step4.shape}")
                
                period_portfolio_risk = step2 + step4
                #print(f"period_portfolio_risk shape: {period_portfolio_risk.shape}")
               # print(f"period_portfolio_risk type: {type(period_portfolio_risk)}")
                
                # 確保結果是標量
                if isinstance(period_portfolio_risk, pd.DataFrame):
                    if period_portfolio_risk.size == 1:
                        period_portfolio_risk = np.sqrt(period_portfolio_risk.iloc[0, 0])
                    else:
                        print(f"Warning: period_portfolio_risk is not a scalar: {period_portfolio_risk}")
                        continue
                elif isinstance(period_portfolio_risk, np.ndarray):
                    if period_portfolio_risk.size == 1:
                        period_portfolio_risk = np.sqrt(period_portfolio_risk.item())
                    else:
                        print(f"Warning: period_portfolio_risk is not a scalar: {period_portfolio_risk}")
                        continue
                else:
                    print(f"Warning: Unexpected type for period_portfolio_risk: {type(period_portfolio_risk)}")
                    continue
                
                forecast_risk.loc[date] = period_portfolio_risk
                
            except Exception as e:
                print(f"Error calculating forecast risk for date {date}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        #print(forecast_risk)
        self.forecast_risk = forecast_risk
        return forecast_risk


    def calculate_forecast_risk2(self):
        if self.resid is None:
            raise ValueError("resid must be calculated before calculating 特定風險")
            
        factor_risk = self.Newey_West_cov
        weights = self.cap_weight.reindex(columns=self.resid.columns)
        resid = self.resid
        portfolio_exposure = self.factors
        forecast_risk = pd.Series(np.nan, index=factor_risk.keys(), name='forecast_risk')
        

            
        for index, date in tqdm(enumerate(forecast_risk.index), desc='計算 forecast risk'):
            try:
                period_weights = weights.loc[date].fillna(0)
                period_portfolio_exposure = portfolio_exposure.loc[date]
                period_factor_risk = factor_risk[date].dropna()
                period_resid = np.diag(resid.loc[date].dropna() ** 2)
                
                if period_factor_risk.empty:
                    print(f"Warning: Empty factor_risk for date {date}")
                    continue
                # 添加維度檢查
                #print(f"Date: {date}")
                #print(f"period_portfolio_exposure shape: {period_portfolio_exposure.shape}")
                #print(f"period_factor_risk shape: {period_factor_risk.shape}")
                #print(f"period_weights shape: {period_weights.shape}")
                #print(f"period_resid shape: {period_resid.shape}")
                
                # 調整維度如果需要
                #period_portfolio_exposure = period_portfolio_exposure.values.reshape(1, -1)
                #period_weights = period_weights.values.reshape(-1, 1)
                
                if period_portfolio_exposure.shape[1] != period_factor_risk.shape[0]:
                   print(f"Dimension mismatch: period_portfolio_exposure columns ({period_portfolio_exposure.shape[1]}) != period_factor_risk rows ({period_factor_risk.shape[0]})")
                   continue
               
                period_portfolio_risk = (period_portfolio_exposure @ period_factor_risk @ period_portfolio_exposure.T) +\
                                        (period_weights.T @ period_resid @ period_weights)
                
                period_portfolio_risk = np.sqrt(period_portfolio_risk[0][0])  # 確保結果是標量
                forecast_risk.loc[date] = period_portfolio_risk
                
            except Exception as e:
                print(f"Error calculating forecast risk for date {date}: {str(e)}")
        
        self.forecast_risk = forecast_risk
        return forecast_risk
        
    def calculate_z_scores(self, q=1):
        """
        計算樣本外z分數，如公式(7.1)所示
        z_t,q = r_t+q / σ_t
        """
        if self.forecast_risk is None:
            raise ValueError("forecast_risk must be calculated before calculating z-scores")
        
        specific_risk = self.forecast_risk
        realized_ret = self.data.pivot(index='date', columns='stocknames', values='ret')
    
        z_scores = []
    
        for t in range(len(specific_risk) - q):
            sigma_t = specific_risk.iloc[t]
            r_tq = realized_ret.iloc[t + q]
    
            z = r_tq / sigma_t
            z_scores.append(z)
        #print(z_scores)
        return pd.DataFrame(z_scores, index=specific_risk.index[:-q], columns=realized_ret.columns)

    def calculate_bias_statistic(self, z_scores, T):
        """
        計算偏差統計量，如公式(7.2)所示
        b_t,T = sqrt(1/(T-1) * sum((z_t,q - z_q)^2))
        """

        bst =  z_scores.rolling(window=5, min_periods=1).std()
        #print(bst)
        return bst
        
    def calculate_bias_statistic2(z_scores, T):
        """
        计算偏差统计量 b_t,T，如公式(7.2)所示：
        b_t,T = sqrt(1/(T-1) * sum((z_s,q - z_q)^2))
    
        参数:
        z_scores : pd.DataFrame, 包含z分数的时间序列数据
        T : int, 测试期间的长度
    
        返回:
        pd.Series: 偏差统计量时间序列
        """
        if len(z_scores) < T:
            raise ValueError(f"z_scores length ({len(z_scores)}) must be at least T ({T})")
    
        bias_statistic = pd.Series(index=z_scores.index[T-1:], dtype='float64')
    
        for t in range(T-1, len(z_scores)):
            window_z_scores = z_scores.iloc[t-T+1:t+1]
            z_q = window_z_scores.mean()
            squared_diff_sum = ((window_z_scores - z_q) ** 2).sum().sum()
            bias_stat = np.sqrt(squared_diff_sum / (T - 1))
            bias_statistic.iloc[t-T+1] = bias_stat  # 确保 bias_stat 是单一标量值
    
        return bias_statistic

    def calculate_confidence_interval(self, T):
        """
        計算95%置信區間，如公式(7.3)所示
        C_T = [1 - sqrt(2/T), 1 + sqrt(2/T)]
        """
        return [1 - np.sqrt(2/T), 1 + np.sqrt(2/T)]

    def winsorize_z_scores(self, z_scores):
        """
        對z分數進行縮尾處理，如公式(7.4)所示
        z_t,q = max(-3, min(+3, z_t,q))
        """
        return np.clip(z_scores, -3, 3)

    def calculate_AD(self, bias_statistic):
        """
        計算AD統計量，如公式(7.5)所示
        AD_t,T = |b_t,T - 1|
        """
        #print(np.abs(bias_statistic - 1))
        return np.abs(bias_statistic - 1)

    def calculate_RAD(self, AD_values, t0, t1):
        """
        計算RAD統計量，如公式(7.6)所示
        RAD_[t_0,t_1] = 1/(t_1-t_0) * sum(AD_t,T)
        """
        return np.mean(AD_values[t0:t1])

    def run_bias_tests(self, window=12):
        """
        執行所有偏誤測試
        """
        print('\n\n==================================執行 bias test ====================================') 

        self.forecast_risk = self.calculate_forecast_risk()
        z_scores = self.calculate_z_scores(q = 1)
        #print(z_scores)
        bias_statistic = self.calculate_bias_statistic(z_scores, window)
        confidence_interval = self.calculate_confidence_interval(window)
        winsorized_z_scores = self.winsorize_z_scores(z_scores)
        robust_bias_statistic = self.calculate_bias_statistic(winsorized_z_scores, window)
        AD = self.calculate_AD(bias_statistic)
        RAD = self.calculate_RAD(AD, 0, len(AD))  # 使用所有可用的AD值

        results = {
            'z_scores': z_scores,
            'bias_statistic': bias_statistic,
            'confidence_interval': confidence_interval,
            'winsorized_z_scores': winsorized_z_scores,
            'robust_bias_statistic': robust_bias_statistic,
            'AD': AD,
            'RAD': RAD
        }

        return results

    def plot_bias_test_results(self, results):
        """
        繪製偏誤測試結果
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(15, 20))

        # 繪製偏誤統計量和置信區間
        axes[0].plot(results['bias_statistic'])
        axes[0].axhline(y=results['confidence_interval'][0], color='r', linestyle='--')
        axes[0].axhline(y=results['confidence_interval'][1], color='r', linestyle='--')
        axes[0].set_title('Bias Statistic with Confidence Interval')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Bias Statistic')

        # 繪製穩健偏誤統計量
        axes[1].plot(results['robust_bias_statistic'])
        axes[1].set_title('Robust Bias Statistic')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Robust Bias Statistic')

        # 繪製AD統計量
        axes[2].plot(results['AD'])
        axes[2].set_title('AD Statistic')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('AD Statistic')

        # 繪製RAD統計量
        axes[3].axhline(y=results['RAD'], color='g', linestyle='-')
        axes[3].set_title('RAD Statistic')
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('RAD Statistic')

        plt.tight_layout()
        plt.show()



class SpecificRisk:
    def __init__(self, mfm, variant='EUE3S'):
        self.mfm = mfm
        self.variant = variant
        self.set_parameters()

    def set_parameters(self):
        params = {
            'EUE3S': {'half_life': 65, 'sample_size': 360},
            'EUE3L': {'half_life': 180, 'sample_size': 360}
        }
        self.half_life = params[self.variant]['half_life']
        self.sample_size = params[self.variant]['sample_size']
        self.lambda_ = 0.5 ** (1 / self.half_life)

    def calculate_specific_returns(self):
        return self.mfm.Un

    def zu(self, specific_returns):
        robust_std = self.calculate_robust_estimator(specific_returns)
        truncated_returns = np.clip(specific_returns, -10*robust_std, 10*robust_std)
        equal_weighted_std = np.std(truncated_returns, axis=0)
        zu = np.abs(equal_weighted_std - robust_std) / robust_std
        return truncated_returns, zu

    def calculate_time_series_forecast(self, specific_returns):
        T = len(specific_returns)
        h = min(T, self.sample_size)
        recent_returns = specific_returns[-h:]
        weights = self.lambda_ ** np.arange(h-1, -1, -1)
        weights = weights / np.sum(weights)
        u_bar = np.mean(recent_returns, axis=0)
        weighted_squared_deviations = weights[:, np.newaxis] * (recent_returns - u_bar)**2
        sdev = np.sqrt(np.sum(weighted_squared_deviations, axis=0))
        return sdev

    def calculate_robust_estimator(self, specific_returns):
        q1, q3 = np.percentile(specific_returns, [25, 75], axis=0)
        a_n = (q3 - q1) / 1.35
        return a_n 

    def calculate_blending_coefficient(self, specific_returns):
        h = len(specific_returns)
        _, z_u = self.zu(specific_returns)
        if h <= 60:
            return np.zeros_like(z_u)
        gamma_1 = np.minimum(1, np.maximum(0, (h - 60) / 120))
        gamma_2 = np.minimum(1, np.maximum(0, np.exp(1 - z_u)))
        gamma = gamma_1 * gamma_2
        return gamma

    def forecast_specific_risk(self):
        specific_returns = self.calculate_specific_returns()
        truncated_returns, zu = self.zu(specific_returns)
        time_series_estimate = self.calculate_time_series_forecast(truncated_returns)
        robust_estimate = self.calculate_robust_estimator(truncated_returns)
        gamma = self.calculate_blending_coefficient(truncated_returns)
        specific_risk = gamma * time_series_estimate + (1 - gamma) * robust_estimate
        return specific_risk, zu

    def structural_std(self):
        structural_std = []
        all_stocknames = set()  # 用於收集所有出現過的股票名稱
        print('===================================逐時計算結構性標準差===================================')       
        for t in tqdm(range(self.mfm.T), desc='橫截面多因子回歸'):
            date = self.mfm.sorted_dates[t]
            data_by_time = self.mfm.data[self.mfm.data['date'] == date.strftime('%Y-%m-%d')]
            
            if data_by_time.empty:
                print(f"警告：日期 {date} 沒有數據，跳過")
                continue
            
            data_by_time = data_by_time.sort_values(by='stocknames')
            all_stocknames.update(data_by_time['stocknames'])  # 添加當前日期的所有股票名稱
            specific_returns = self.calculate_specific_returns()
            
            if date not in specific_returns.index:
                print(f"警告：日期 {date} 在特異性收益中不存在，跳過")
                continue
            
            time_series_forecast = self.calculate_time_series_forecast(specific_returns.loc[:date])
            
            if np.isnan(time_series_forecast).all():
                print(f"警告：日期 {date} 的時間序列預測全為 NaN，跳過")
                continue
            
            try:
                cs = CrossSection(data_by_time, log_forecast = np.log(np.where(time_series_forecast == 0, np.nextafter(0, 1), time_series_forecast))
)
                _, log_structural_std_t = cs.reg3()
                structural_std_t = np.exp(log_structural_std_t)
                structural_std.append(structural_std_t.flatten())  # 將結果展平為一維陣列
            except Exception as e:
                print(f"處理日期 {date} 時發生錯誤: {str(e)}")
                continue
        
        if not structural_std:
            raise ValueError("沒有成功計算出任何結構性標準差")
        
        # 將結果轉換為二維 NumPy 陣列
        structural_std_array = np.array(structural_std)
        
        # 檢查並調整形狀
        if structural_std_array.ndim == 3:
            structural_std_array = structural_std_array.reshape(structural_std_array.shape[0], -1)
        
        # 使用所有出現過的股票名稱作為列名
        all_stocknames = sorted(list(all_stocknames))
        
        # 創建一個全為 NaN 的 DataFrame
        structural_std_df = pd.DataFrame(np.nan, 
                                         index=self.mfm.sorted_dates[:len(structural_std)], 
                                         columns=all_stocknames)
        
        # 填充數據
        for i, date in enumerate(structural_std_df.index):
            data_by_time = self.mfm.data[self.mfm.data['date'] == date.strftime('%Y-%m-%d')]
            stocknames = data_by_time['stocknames']
            structural_std_df.loc[date, stocknames] = structural_std_array[i][:len(stocknames)]
        
        return structural_std_df

    def run(self):
        time_series_forecast, zu = self.forecast_specific_risk()
        structural_forecast = self.structural_std()
        scale_factor = np.sqrt(np.mean(time_series_forecast**2) / np.mean(structural_forecast**2))
        final_forecast = scale_factor * structural_forecast
        self.mfm.resid = final_forecast
        return final_forecast

data = pd.read_csv('C:\\Users\\385225\\Documents\\GitHub\\Barra\\data\\cleaned_data2.csv')
orgdata = data

# 篩選出2018年5月15日之後的所有行
data = data[data['date'] >= '2018-05-15']
data1 = data[data['date'] >= '2023-10-01']

model = MFM(data1, 9, 31)
(factor_ret, specific_ret, R2, Un) = model.reg_by_time()

#
nw_cov_ls = model.Newey_West_by_time(q = 15, tao = 90, h = 252)                 #Newey_West调整
#
#specific_risk_model = SpecificRisk(model, variant='EUE3S')  # 或 'EUE3L'
specific_risk = model.calculate_specific_risk(variant='EUE3S')

# 運行偏誤測試
bias_test_results = model.run_bias_tests(window=12)
#%%
# 繪製偏誤測試結果
model.plot_bias_test_results(bias_test_results)

#%%
forcast_risk = model.forecast_risk

#%%

er_cov_ls = model.eigen_risk_adj_by_time(M = 100, scale_coef = 1.4)    #特征风险调整
vr_cov_ls, lamb = model.vol_regime_adj_by_time(tao = 42) 


#%%


#%%

factor_value_df = factor_ret.add(1).cumprod().dropna(axis=0)

plt.figure(figsize=(20, 12))
plt.plot(factor_value_df) #21, 9
plt.title('Net Value of Factors')
plt.ylabel('Net Value')
plt.xlabel('Time')
plt.legend(factor_value_df.columns)
#plt.yticks(range(20))
# plt.ylim(0, 3)
plt.show()

#%%
R2 = R2[R2<1]
plt.figure(figsize=(20,12))
plt.plot(R2.rolling(window=250).mean(), label= 'Cap Weights')

plt.legend()
plt.grid(True)
plt.show()


#%%
'''
import yfinance as yf

P0050 = yf.download('0050.TW', start = factor_value_df.index[0], end= factor_value_df.index[-1])['Adj Close'].pct_change().fillna(0).add(1).cumprod()
PTWII = yf.download('^TWII', start = factor_value_df.index[0], end= factor_value_df.index[-1])['Adj Close'].pct_change().fillna(0).add(1).cumprod()

plt.figure(figsize=(20, 12))
plt.plot(factor_value_df.iloc[:,0], label = 'Barra')
plt.plot(P0050, label =  '0050')
plt.plot(PTWII, label =  'TWII - PriceIndex')
plt.title('Net Value of Factors')
plt.ylabel('Net Value')
plt.xlabel('Time')
plt.yscale('log')
plt.legend()
plt.show()

value_holding = factor_value_df.iloc[-1,:].sort_values()




#%%


factor_returns = factor_ret.iloc[:,-9:].dropna(axis=0)#.drop(columns = ['Volatility'])

portfolio_returns = pd.Series(index = factor_returns.index)
for index, date in enumerate(factor_returns.index):
    if index == len(factor_returns.index)-1:
        continue
    factor_daily_return = factor_returns.loc[date].sort_values(ascending=False)
    daily_long_factor = factor_daily_return.index[0]
    daily_short_factor = factor_daily_return.index[-1]
    portfolio_daily_return = factor_returns.iloc[index+1][daily_long_factor] - factor_returns.iloc[index+1][daily_short_factor]
    portfolio_returns.loc[date] = portfolio_daily_return

portfolio_value = portfolio_returns.add(1).cumprod()

plt.figure(figsize=(20, 12))
#plt.plot(factor_value_df['DSTD_65_23'], label = 'DSTD_65_23')
#plt.plot(factor_value_df['CMRA_12_0'], label = 'CMRA_12_0')
# plt.plot(factor_value_df.iloc[:,-21:].drop(columns = ['DSTD_65_23']), label = factor_value_df.iloc[:,-21:].drop(columns = ['DSTD_65_23']).columns)
plt.plot(portfolio_value, label = 'Factor Momentum')
plt.title('Net Value of Factors')
plt.ylabel('Net Value')
plt.xlabel('Time')
plt.legend()
plt.yscale('log')
# plt.legend(factor_value_df.iloc[:,-9:].columns)
plt.show()

'''


