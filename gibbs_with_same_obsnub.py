import numpy as np
import scipy as sp
from pandas import DataFrame
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv

from collections import deque
import pdb

class DataGenerator:
    def __init__(self): #定义超参数
        self.cust_Num = 90
        self.class_Num = 3
        self.observationInGroup = 10
        self.sigma2 = 1
        self.v_alpha = 2
        self.v_beta = 2
        self.betaNumber = 15
        self.MuNonZeros = [
            [1.3,1.5,0.7,0.9,1.0,0.4],
            [0,0,0,16.2,14.0,15.0,10.08,12.3,15],
			[-15.1,-17,-12]
        ]
        self.Mu = np.zeros((self.class_Num,self.betaNumber))
        self.Y,self.X,self.epsilon = self._genData()

    def _genData(self):
        #generate mu
        # true mu array  (3,20)
        for i in range(self.class_Num): 
            self.Mu[i]=np.concatenate([self.MuNonZeros[i],np.zeros(self.betaNumber-len(self.MuNonZeros[i]))])

        #generate H H array (90,1)
        self.H=np.repeat(range(self.class_Num),self.cust_Num/self.class_Num)
 
        #generate v alpha = 2, v_beta = 0.01  产生的数大概位于1附近 
        self.v = sp.stats.invgamma(self.v_alpha,scale=self.v_beta).rvs(1000).mean()

        # generate true beta 
        # beta矩阵是一个90*20的数组，每行即beta[i]是每个顾客的beta值
        # beta_real.shape=(90,20)  beta_real[i].shape = (20,1)
        self.beta_real = np.zeros((self.cust_Num,self.betaNumber)) 
        for i in range(len(self.H)):
            h = self.H[i]
            sigma2_mu=self.v*np.eye(self.betaNumber)
            self.beta_real[i] = np.random.multivariate_normal(self.Mu[h],sigma2_mu,size = 100).mean(axis=0).reshape(1,-1)

        #generate epsilon 
        #每行是每个顾客的16个观测
        #epsilon[i]是(16,1)的数组
        self.epsilon = np.random.normal(0,self.sigma2,(self.cust_Num,self.observationInGroup))

        #generate X
        # X (90,16,20)数组 X[i].shape =(16,20 ) 第i个顾客16个观测值 20个变量的信息
        #self.X=np.random.normal(loc = 0 ,scale =1,size=(self.cust_Num,self.observationInGroup,self.betaNumber))
        self.X=np.random.randint(1,10,size=(self.cust_Num,self.observationInGroup,self.betaNumber))

        #generate Y
        # Y.shape = (90,16) Y[10].shape = (16,1)
        self.Y = np.random.randn(self.cust_Num,self.observationInGroup)
        for i in range(self.cust_Num):
            self.Y[i] = self.X[i].dot(self.beta_real[i])+self.epsilon[i]

        return self.Y,self.X,self.epsilon

    def getData(self):
        return self.Y,self.X,self.epsilon

    def getShape(self):
        return self.cust_Num,self.betaNumber,self.observationInGroup

    def getclassNum(self):
        return self.class_Num
   
    def getTrueBeta(self):
        return self.beta_real

class Simulation:
    def __init__(self,
                v_alpha = 2,
                v_beta = 2,
                tau2_alpha = 2,
                tau2_beta = 0.5,
                w = 0.5,
                data = None
                #K = None
                ):
        #设置超参数 将class DataGenerator 里面的参数继承下来
        self.v_alpha = v_alpha
        self.v_beta = v_beta
        self.tau2_alpha = tau2_alpha
        self.tau2_beta = tau2_beta
        self.w = w
        if data is None:
            self.data = DataGenerator()
            self.Y,self.X ,self.epsilon_real = self.data.getData()
            self.custNum,self.betaNumber,self.obs=self.data.getShape()
            self.true_beta = self.data.getTrueBeta()
        else:
            self.Y , self.X = data
            self.custNum = self.Y.shape[0] # 总的顾客数
            self.betaNumber = self.X.shape[2] # 总的变量数

        
        

    def initialization(self):#生成初始数据
        #  if K is None:
        #     self.classNum = self.data.getclassNum()
        # else:
            #self.classNum = K
        # tau2 初始值
        self.tau2_0 = sp.stats.invgamma(self.tau2_alpha,scale=self.tau2_beta).rvs(1000).mean()
        # v 初始值
        self.v_0 = sp.stats.invgamma(self.v_alpha,scale=self.v_beta).rvs(1000).mean()
        # H pro 初始值 Hpro.shape=(90,3) Hpro[i].shape=(3,1)
        self.Hpro_0 = (1/self.classNum)*np.ones((self.custNum,self.classNum))
        #H 初始化 H.shape=(90,1)
        self.H_0 = np.zeros(self.custNum,dtype=int)
        for i in range(self.custNum):
            prob = self.Hpro_0[i]
            self.H_0[i] = np.random.choice(range(self.classNum),size=1,p=prob)
        # Z K*J个对应于mu的初始化  (3,20) Z_0[i].shape=(20,1) z_0[i,j] 取单个元素
        self.Z_0 = sp.stats.bernoulli(self.w).rvs((self.classNum,self.betaNumber))
        # mu 的初始值 用基于H分类的beta的估计参数
        self.mu_0 = np.zeros((self.classNum,self.betaNumber))
        for idx in range(self.classNum):
            index_for_reg = np.argwhere(self.H_0==idx).flatten()
            #将(X,16)二维数组 降为一维数组 (X*16) 
            #np.ravel(data ) 默认按行降 np.ravel(data,'F') 按列降 
            y_reg = np.ravel(self.Y[index_for_reg])  
           # ind = len(index_for_reg)*self.obs
            x_try = self.X.copy() #复制不修改self.X
            x_blong = x_try[index_for_reg] #找到属于某组的所有x
            ind = np.sum(i.shape[0] for i in x_blong)
            # 将三维数组X 转换成二维 维度是(len(indx)*16,20) 与y相对应
            x_reg = np.reshape(x_blong,(ind,self.betaNumber)) 
            self.mu_0[idx] = inv(x_reg.T.dot(x_reg)).dot(x_reg.T).dot(y_reg)


        # beta 的初始值 生成基于mu v的正态分布
        self.beta_0 = np.zeros((self.custNum,self.betaNumber)) 
        for it in range(self.custNum):
            beta_belong = self.H_0[it]
            mu_forbeta = self.mu_0[beta_belong]
            sigma_for_beta = self.v_0*np.eye(self.betaNumber)
            self.beta_0[it] = np.random.multivariate_normal(mu_forbeta,sigma_for_beta,size=100).mean(axis=0).reshape(1,-1)

        # 生成历史数据记录的deque
        self.MAXLEN = 100
        self.beta_His = deque(maxlen=self.MAXLEN)
        self.tau2_His = deque(maxlen = self.MAXLEN)
        self.mu_His = deque(maxlen = self.MAXLEN )
        self.H_His = deque(maxlen = self.MAXLEN)
        self.v_His = deque(maxlen = self.MAXLEN)
        self.Z_His = deque(maxlen = self.MAXLEN)
        self.Hpro_His = deque(maxlen = self.MAXLEN)

        self.beta_His.append(self.beta_0)
        self.tau2_His.append(self.tau2_0)
        self.mu_His.append(self.mu_0)
        self.H_His.append(self.H_0)
        self.v_His.append(self.v_0)
        self.Z_His.append(self.Z_0)
        self.Hpro_His.append(self.Hpro_0)

    def getK(self):
        return self.classNum
    
    def genZNew(self):
        newZ = self.Z_His[-1].copy()
        vLast = self.v_His[-1]
        tau2Last = self.tau2_His[-1]
        betaLast = self.beta_His[-1]
        HLast = self.H_His[-1]
        w = self.w
        for k in range(self.classNum):
            ind_for_beta = np.argwhere(HLast==k).flatten()
            m_for_class = np.sum(HLast==k)
            for j in range(self.betaNumber):
                beta_sum2 = ((betaLast[ind_for_beta,j]).sum())**2
                log_R = np.log(w/(1-w))+(1/2)*np.log(vLast/(vLast+m_for_class*tau2Last))+(beta_sum2*tau2Last)/(2*vLast*(vLast+m_for_class*tau2Last))
                if log_R > 10:
                    newZ[k,j] = 1
                else:
                    R = np.exp(log_R)
                    newZ[k,j] = sp.stats.bernoulli(R/(1+R)).rvs()
        return newZ

    def genMuNew(self):
        newMu = self.mu_His[-1].copy()
        ZLast = self.Z_His[-1]
        tau2Last = self.tau2_His[-1]
        vLast = self.v_His[-1]
        HLast = self.H_His[-1]
        betaLast = self.beta_His[-1]
        for k in range(self.classNum):
            ind_for_beta = np.argwhere(HLast==k).flatten()
            m_for_class = np.sum(HLast==k)
            for j in range(self.betaNumber):
                beta_sum = (betaLast[ind_for_beta,j]).sum()
                u_for_updt = (tau2Last*beta_sum)/(vLast+tau2Last*m_for_class)
                sigma_updt = 1/(1/tau2Last+m_for_class/vLast)
                newMu[k,j] = np.random.normal(loc=u_for_updt,scale=sigma_updt,size=1)
        newMu = np.multiply(newMu,ZLast)
        #print(newMu)
        return newMu

    def genBetaNew(self):
        Newbeta = self.beta_His[-1].copy()
        vLast = self.v_His[-1]
        HLast = self.H_His[-1]
        muLast = self.mu_His[-1]
        y_for_rex = self.Y.copy()
        x_for = self.X.copy()
        for i in range(self.custNum):
            k_belong = HLast[i]
            for j in range(self.betaNumber):
                x_rex = x_for[i].copy()
                x_minus = x_for[i][:,j]
                x_rex[:,j] = 0
                y_minus = y_for_rex[i]-x_rex.dot(Newbeta[i])
                S_n = 1/(1/vLast+x_minus.T.dot(x_minus))
                mu_n = S_n*(muLast[k_belong,j]/vLast+x_minus.T.dot(y_minus))
                #new_S_n = max(1,S_n)
                Newbeta[i,j] = np.random.normal(loc=mu_n,scale=S_n,size=1)
        return Newbeta
        
    def genVNew(self):
        muLast = self.mu_His[-1]
        betaLast = self.beta_His[-1]
        HLast = self.H_His[-1]
        mu_extend = muLast[HLast]
        beta_mu_sum = ((betaLast-mu_extend)**2).sum()
        t1_plus = self.custNum*self.betaNumber
        #t1_plus = np.sum()
        t2_plus = beta_mu_sum/2
        newV = sp.stats.invgamma(self.v_alpha+t1_plus,scale=self.v_beta+t2_plus).rvs()
        return newV

    def genTau2New(self):
        ZLast = self.Z_His[-1]
        muLast = self.mu_His[-1]
        mu_2 = muLast**2
        mu_time_z = np.multiply(mu_2,ZLast)
        sum_mu2 = np.sum(mu_time_z)
        sum_z = np.sum(ZLast)
        s1_plus = sum_z/2
        s2_plus = sum_mu2/2
        newtau2 = sp.stats.invgamma(self.tau2_alpha+s1_plus,scale=self.tau2_beta+s2_plus).rvs()
        return newtau2
    
    def genHproNew(self):
        newHpro = np.zeros((self.custNum,self.classNum),dtype= float)
        non_exp = np.zeros((self.custNum,self.classNum),dtype= float)
        sum_Hpro = np.zeros((self.custNum,self.classNum),dtype = float)
        betaLast = self.beta_His[-1]
        muLast = self.mu_His[-1]
        vLast = self.v_His[-1]
        for i in range(self.custNum):
            for j in range(self.classNum):
                b_mu_Sum = betaLast[i]-muLast[j]
                sum2 = (b_mu_Sum**2).sum()
                non_exp[i,j] = -1*sum2/(2*vLast)
            for k in range(self.classNum):
                sum_exp = np.exp(non_exp[i]-non_exp[i,k]).sum()
                newHpro[i,k] = 1/sum_exp
        #print(non_exp)
        #print(newHpro.sum(1))        
        for td in range(self.classNum):
            sum_Hpro[:,td] = newHpro.sum(1)
        new_Hpro = np.multiply(newHpro,1/sum_Hpro)
        #print(new_Hpro)
        return new_Hpro

    def genHNew(self):
        newH =  np.zeros(self.custNum,dtype=int)
        HproLast = self.Hpro_His[-1]
        for i in range(self.custNum):
            pt = HproLast[i]
            ct = np.arange(self.classNum)
            newH[i] = np.random.choice(ct,size=1,replace=True,p=pt)
        return newH
    
    
    def true_beta_mins_beta(self):
        self.extraction = self.true_beta-self.beta_all_estimation
        print(self.extraction)

    def simulation(self):
        self.likelihood_f = []
        for ite in range(1500):
            zNew = self.genZNew()
            self.Z_His.append(zNew)

            muNew = self.genMuNew()
            self.mu_His.append(muNew)

            betaNew = self.genBetaNew()
            self.beta_His.append(betaNew)

            HproNew = self.genHproNew()
            self.Hpro_His.append(HproNew)

            Hnew = self.genHNew()
            self.H_His.append(Hnew)

            vNew = self.genVNew()
            self.v_His.append(vNew)
            
            tau2New = self.genTau2New()
            self.tau2_His.append(tau2New)

            #print(ite,tau2New,vNew)
            #if ite>1490:
            #    print(np.round(HproNew,3))
            #    print(np.round(muNew,3))
            #    print(np.round(betaNew,3))
            #    print(tau2New,vNew)
            #    print(zNew) 
            #    print(Hnew)   

            sum_likelihood=0.0
            reg_y = self.Y.copy()
            reg_x = self.X.copy()
            for ti in range(self.custNum):
                y_ti_reg = reg_y[ti]
                x_ti_reg = reg_x[ti]
                beta_ti_reg = betaNew[ti]
                likely = y_ti_reg-x_ti_reg.dot(beta_ti_reg)
                sum_likely = np.sum(likely**2)
                sum_likelihood += sum_likely
            exp_for_l = -1*sum_likelihood/100
            self.likelihood_f.append(exp_for_l)
    
    def plotLikelihood(self):
        plt.plot(range(1490),self.likelihood_f[10:])
        plt.title('likelihood curve')
        plt.savefig(r'C:\Users\SongGe\jiaoben_Jupyter\beta{}.jpg'.format(self.classNum))
        #plt.show()

    def getRandom(self,K):
        self.classNum = K
        self.initialization()
        self.simulation()
        #return self.betaHis[-1],self.tau2His[-1],self.muHis[-1],self.HproHis[-1].T,self.HHis[-1],self.vHis[-1],self.zhis[-1]
        #return self.mu_His[-1],self.Z_His[-1],self.beta_His[-1],self.Hpro_His[-1]
    
    def get_estimation_after(self):
        self.beta_all_estimation = np.concatenate([k[np.newaxis,:]for k in self.beta_His]).mean(0)
        self.tau2_all_estimation = np.mean(self.tau2_His)
        #self.mu_all_estimation = np.concatenate([k[np.newaxis,:]for k in self.mu_His]).mean(0)
        #self.H_all_estimation = np.median(np.concatenate([k[np.newaxis,:]for k in self.H_His]),0)
        self.H_all_estimation = self.H_His[-1]
        self.v_all_estimation = np.mean(self.v_His)
        self.Z_all_estimation = np.median(np.concatenate([k[np.newaxis,:]for k in self.Z_His]),0)
        cum_Z = np.concatenate([k[np.newaxis,:]for k in self.Z_His])
        cum_mu = np.concatenate([k[np.newaxis,:]for k in self.mu_His])
        self.mu_all_estimation = np.multiply(cum_Z,cum_mu).mean(0)
        self.Hpro_all_estimation = np.concatenate([k[np.newaxis,:]for k in self.Hpro_His]).mean(0)
        return self.beta_all_estimation,self.tau2_all_estimation,self.mu_all_estimation,self.H_all_estimation,\
        self.v_all_estimation,self.Z_all_estimation,self.Hpro_all_estimation
    
    def get_Estimation(self,*args): # 可以得到Z tau2 v H beta mu Hpro 输入对应参数的估计值
        if len(args)==1:
            value = np.mean(args)
        elif len(args)==2: #有两个参数时，如H-his 求众数Hi
            name = args[0]
            i = args[1]
            value_matix = np.concatenate([j[np.newaxis,:]for j in name])
            counts = np.bincount(value_matix[:,i-1])
        #返回众数
            value = np.argmax(counts)
        elif len(args) == 3: #当三个参数时，如beta-his 求beta_ij 均值
            name = args[0]
            i = args[1]
            j = args[2]
            value_matix = np.concatenate([k[np.newaxis,:]for k in name]).mean(0)
            value = value_matix[i-1,j-1]
        else: # Z mu i j的参数输入形式
            Z = args[0]
            name = args[1]
            i = args[2]
            j = args[3]
            Z_matix = np.concatenate([k[np.newaxis,:]for k in Z]).mean(0)
            if Z_matix[i-1,j-1]<0.5:
                value = 0
            else:
            ### plan A ；mean(mu) Under all Z
                value_matix = np.concatenate([k[np.newaxis,:]for k in name]).mean(0)
                value = value_matix[i-1,j-1]
                ### plan B : mean(mu) under Z ≠ 0
                #cum_Z = np.concatenate([k[np.newaxis,:]for k in Z])
                #cum_value = np.concatenate([k[np.newaxis,:]for k in name])
                #value_matix = np.multiply(cum_Z,cum_value).mean(0)
                #value = value_matix[i-1,j-1]
        return value
    
    def get_Distribution(self,*args): # # 可以得到Z tau2 v H beta mu Hpro 输入对应参数的分布图像
        if len(args)==1:
            value = args
            plt.hist(value,bins=100)
            plt.show()
        elif len(args)==2:
            name = args[0]
            i = args[1]
            value_matix = np.concatenate([j[np.newaxis,:]for j in name])
            value = value_matix[:,i-1]
            plt.hist(value ,bins=100)
            plt.show()
        else:
            name = args[0]
            i = args[1]
            j = args[2]
            value_matix = np.concatenate([k[np.newaxis,:]for k in name])
            value = value_matix[:,i-1,j-1]
            plt.hist(value,bins = 100)
            plt.show()

    def get_true_beta_minus_lastbeta(self):
        DataFrame(self.true_beta-self.beta_His[-1]).to_csv('true_minus_last_{}.csv'.format(self.classNum))

    def getBIC(self):
        beta_estimator = self.beta_all_estimation
        mu_estimator = self.mu_all_estimation
        self.res_left = np.zeros((self.custNum,self.obs))
        sum_constance = 0
        sumlikelihood= 0
        realsumlikelihodd = 0
        y_all = self.Y.copy()
        x_all = self.X.copy()
        real_beta = self.true_beta
        #self.res2 = np.zeros((self.custNum,self.obs))
        for cust in range(self.custNum):
                y_cust_reg = y_all[cust]
                x_cust_reg = x_all[cust]
                beta_cust_reg = beta_estimator[cust]
                likely = y_cust_reg - x_cust_reg.dot(beta_cust_reg)
                self.res_left[cust] = likely 
                #self.res2[cust] = likely
                constance = len(y_cust_reg)*np.log(2*np.pi*likely.var())
                real_likeli =  y_cust_reg-x_cust_reg.dot(real_beta[cust])
                #sum_likely = np.sum(likely**2)    # 求overall σ2 的estimator
                #针对每个顾客 用 res2 / res2(i).var()  计算每个顾客的 σ2 的估计
                sum_likely = np.sum(likely**2)/likely.var()
                #sum_likely = np.sum(likely**2)
                real_sum_likely = np.sum(real_likeli**2)/real_likeli.var()
                sumlikelihood += sum_likely
                realsumlikelihodd += real_sum_likely
                sum_constance += constance
                print('cust var :',likely.var())
                #print('{0} estimated likely :{1}'.format(cust,sum_likely))
                #print('{0} real likely :{1}'.format(cust,real_sum_likely))
        sum_n = np.sum(len(i) for i in self.Y) # observation 求和
        #sum_n = self.custNum
        # beta_ij<0.05 = 0 
        for i in range(self.custNum):
            mask = np.argwhere(beta_estimator[i]<0.01).flatten()
            beta_estimator[i][mask] = 0
        beta_var = np.sum(np.sum(i!= 0) for i in beta_estimator) 
        #beta_var = self.custNum*self.betaNumber
        mu_var = np.sum(np.sum(i!= 0) for i in mu_estimator)
        varible_k = beta_var+mu_var
        # 计算 overall 的 sum(res2)/res.var()
        BIC  = sum_constance+sumlikelihood+np.log(sum_n)*varible_k
        realBIC = self.custNum*self.obs*np.log(2*np.pi)+realsumlikelihodd+np.log(sum_n)*varible_k
        print('sum_constance is {}'.format(sum_constance))
        print('real_sum_constance is {}'.format(self.custNum*self.obs*np.log(2*np.pi)))
        print('BIC is {}'.format(BIC))
        print('sum likelihood divided by sigma is {}'.format(sumlikelihood))
        print('realBIC is {}'.format(realBIC))
        print('real sum likelihood is {}'.format(realsumlikelihodd))

    def save_estimator_to_csv(self):
        df = DataFrame(self.beta_all_estimation)
        # df['H'] = self.H_all_estimation
        df.to_csv(r'estimator\beta_and_H_{}.csv'.format(self.classNum))
        # DataFrame(self.mu_all_estimation).to_csv(r'estimator\mu_{}.csv'.format(self.classNum))
        # #DataFrame(self.H_all_estimation).to_csv(r'estimator\H_{}.csv'.format(self.classNum))
        # DataFrame(self.Z_all_estimation).to_csv(r'estimator\Z_{}.csv'.format(self.classNum))
        # DataFrame(self.Hpro_all_estimation).to_csv(r'estimator\Hpro_{}.csv'.format(self.classNum))
        DataFrame(self.extraction).to_csv(r'estimator\ture_minus_beta_{}.csv'.format(self.classNum))
        #DataFrame(self.res_left).to_csv(r'estimator\residual_{}.csv'.format(self.classNum))
        DataFrame(self.true_beta).to_csv(r'estimator\true_beta_{}.csv'.format(self.classNum))
if __name__ == "__main__":
    # dataloader
    #data = dataloader('.csv')
    #for k in range(2,7):
    sim = Simulation()
    for ttt in range(1,8):
        sim.getRandom(K=ttt)  
        print(sim.getK())
        sim.get_estimation_after()
        sim.true_beta_mins_beta()
        sim.save_estimator_to_csv()
        sim.getBIC()