# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:26:11 2024

@author: 385225
"""

import os
os.chdir('C:\\Users\\385225\\Documents\\GitHub\\Barra')



from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



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
    def __init__(self, data, robust_ret=None, specific_ret = None ):
        #print("Initializing CrossSection...")
        #print("Data Columns:", data.columns)
        
        # 拆分数据
        self.date = data['date'].values[0]
        self.stocknames = data['stocknames'].values
        self.capital = data['capital'].values
        self.ret = data['ret'].values
        self.robust_ret =  robust_ret
        self.specific_ret = specific_ret
        
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
        
        #self.W = np.sqrt(self.capital) / sum(np.sqrt(self.capital))
        self.W = np.clip(np.sqrt(self.capital) / sum(np.sqrt(self.capital)), None, np.quantile(np.sqrt(self.capital) / sum(np.sqrt(self.capital)), 0.95))
        
        #print(f'Cross Section Regression, Date: {self.date}, {self.N} Stocks, {self.P} Industry Factors, {self.Q} Style Factors')

    
    def reg1(self):
        '''
        多因子模型求解
        '''
        
        W = np.diag(self.W)
        
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
            国家因子：第一行表示国家因子纯因子组合在各个因子上的暴露，
                     国家因子纯因子组合在国家因子上暴露为1，在风格因子上暴露也为0
                     但是在行业因子上不为0（暴露恰好为行业市值权重）
                     而在行业中性限制下国家因子纯因子组合在行业因子暴露上获得的收益恰好为0
                     从而国家因子纯因子组合的收益就是国家因子的收益 f_c
            
            行业因子：第2行-第2+P行表示行业纯因子组合在各个因子上的暴露
                     行业因子纯因子组合在国家因子上暴露为0，在风格因子上暴露也为0
                     但是在各个行业因子上不为0
                     这里所谓的行业纯因子组合是指 做多行业纯因子，同时做空国家纯因子组合
                     为了得到真正的行业纯因子，应该是第一行+第2行，获得的收益是国家因子收益+该行业的纯因子收益
                     这里算出来的行业因子纯因子组合的收益应该理解为行业因子的收益与国家因子的相对收益
            
            风格因子：在风格因子上暴露为1，其他因子上暴露为0，收益为风格纯因子的收益
            '''
            
        else:
            #求解多因子模型
            factors = np.matrix(np.hstack([self.country_factors, self.style_factors]))
            pure_factor_portfolio_weight = np.linalg.inv(factors.T @ W @ factors) @ factors.T @ W    #純因子組合權重

        
        factor_ret = pure_factor_portfolio_weight @ self.ret                        #純因子收益
    
        factor_ret = np.array(factor_ret)[0]
        
        pure_factor_portfolio_exposure = pure_factor_portfolio_weight @ factors     #纯因子组合在各个因子上的暴露
        specific_ret = self.ret - np.array(factors @ factor_ret.T)[0]               #個股特異收益率 (Un)
        #R2 = 1 - np.var(specific_ret) / np.var(self.ret)                            #R square
        
        #return((factor_ret, specific_ret, pure_factor_portfolio_exposure, R2))
        return((factor_ret, specific_ret, pure_factor_portfolio_exposure))

    def reg2(self):
        '''
        多因子模型求解
        '''
        
        W = np.diag(self.W)
        
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
        
        #print(type(self.robust_ret))
        factor_ret = pure_factor_portfolio_weight @ self.robust_ret                        #純因子收益
        
        factor_ret = np.array(factor_ret)[0]
        #print('ok')
        pure_factor_portfolio_exposure = pure_factor_portfolio_weight @ factors     #纯因子组合在各个因子上的暴露
        
        #specific_ret = self.ret - np.array(factors @ factor_ret.T)[0]               #個股特異收益率 (Un)
        R2 = 1 - np.var(self.specific_ret) / np.var(self.ret)                            #R square
        #print('alright')
        #return((factor_ret, specific_ret, pure_factor_portfolio_exposure, R2))
        return((factor_ret,  pure_factor_portfolio_exposure, R2))
    
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
        
    def reg_by_time(self):
        '''
        逐时间点进行横截面多因子回归
        '''
        factor_ret = []
        R2 = []
        specific_ret = []
        specific_ret_list = [] #********************
        
     #   print('===================================逐時第一次横截面多因子回歸===================================')       
    #    for t in range(self.T):
   #         data_by_time = self.data.iloc[self.dates == self.sorted_dates[t],:]
  #          data_by_time = data_by_time.sort_values(by = 'stocknames')
            
 #           cs = CrossSection(data_by_time)
#            factor_ret_t, specific_ret_t, _ , R2_t = cs.reg()
            
        #    factor_ret.append(factor_ret_t)
            #注意：每个截面上股票池可能不同
       #     specific_ret.append(pd.DataFrame([specific_ret_t], columns = cs.stocknames, index = [self.sorted_dates[t]]))
      #      specific_ret_list.append(pd.DataFrame(specific_ret_t, index=cs.stocknames, columns=[self.sorted_dates[t]]).T) #*****************
     #       R2.append(R2_t)
    #        self.last_capital = cs.capital
   #     factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)      
   
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
        #print("出jump迴圈")
        #print(return_jump_df)
        robust_ret_df = return_jump_df
        
        
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
        for t in tqdm(range(self.T), desc='第二次横截面多因子回歸'):
            data_by_time = self.data.iloc[self.dates == self.sorted_dates[t],:]
            data_by_time = data_by_time.sort_values(by = 'stocknames')
            
            cs = CrossSection(data_by_time, robust_ret=data_by_time['robust_ret'].values, specific_ret = self.specific_ret )
            factor_ret_t, _ , R2_t = cs.reg2()
            
            factor_ret.append(factor_ret_t)
            #注意：每个截面上股票池可能不同
            #specific_ret.append(pd.DataFrame([specific_ret_t], columns = cs.stocknames, index = [self.sorted_dates[t]]))
            #specific_ret_list.append(pd.DataFrame(specific_ret_t, index=cs.stocknames, columns=[self.sorted_dates[t]]).T) #*****************
            R2.append(R2_t)
            self.last_capital = cs.capital
            
        factor_ret = pd.DataFrame(factor_ret, columns = self.columns, index = self.sorted_dates)
        R2 = pd.DataFrame(R2, columns = ['R2'], index = self.sorted_dates)
        #Un = pd.concat(specific_ret_list).sort_index()               #***************************************
        
        print('\n===================================完成===================================')
     #*******************************************************************************************************   
        self.factor_ret = factor_ret                                               #因子收益
        
        self.Un = Un                                                               #特异性收益
        self.R2 = R2                                                               #R2
        return((factor_ret, specific_ret, R2, Un))


 #   def Jumps_adjust(self):
        


    def Newey_West_by_time(self, q = 2, tao = 252):
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
                Newey_West_cov.append(Newey_West(self.factor_ret[:t], q, tao))
            except:
                Newey_West_cov.append(pd.DataFrame())
            
            progressbar(t, self.T, '   date: ' + str(self.sorted_dates[t-1])[:10])
        
        self.Newey_West_cov = Newey_West_cov
        return(Newey_West_cov)
    
    
    
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

    




#%%
data = pd.read_csv('C:\\Users\\385225\\Documents\\GitHub\\Barra\\data\\cleaned_data.csv')
orgdata = data
# 篩選出2018年5月15日之後的所有行
data = data[data['date'] >= '2018-05-15']
data1 = data[data['date'] >= '2023-12-01']
#print("Data Columns:", data.columns)
#%%

model = MFM(data1, 9, 31)
(factor_ret, specific_ret, R2, Un) = model.reg_by_time()


#%%

nw_cov_ls = model.Newey_West_by_time(q = 2, tao = 252)                 #Newey_West调整

#%%

er_cov_ls = model.eigen_risk_adj_by_time(M = 100, scale_coef = 1.4)    #特征风险调整
vr_cov_ls, lamb = model.vol_regime_adj_by_time(tao = 42) 


#%%

model.factor_ret.to_csv('factor_ret.csv')

#%%

import matplotlib.pyplot as plt



plt.figure(figsize=(20,12))
plt.plot(R2.rolling(window=250).mean(), label= 'Cap Weights')

plt.legend()
plt.grid(True)
plt.show()


