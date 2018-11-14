# gibbs_program
# Gibbs Sampling Program

The algorithm is based on  **Bayesian Collaborative Model** aimed to fix the problem that the regression parameter is larger than the observation number and which could get the Estimation of parameters   ![](https://github.com/songge529/gibbs_program/raw/master/letter/Z.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/mu.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/beta.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/H.gif), Simultaneously.Because the data is real Pharmaceutical production data, different  medicines have different batches. Each batch is response variable, each batch has different production records. Therefore, we assumed that some different batch maybe belong to same class with some attributes.   

## Parameter Description: 

![](https://github.com/songge529/gibbs_program/raw/master/letter/Z.gif) - Variable selection with 0 or 1 ![](https://github.com/songge529/gibbs_program/raw/master/letter/Z_ij.gif) = 1 means ![](https://github.com/songge529/gibbs_program/raw/master/letter/mu_ij.gif) != 0 

![](https://github.com/songge529/gibbs_program/raw/master/letter/mu.gif) - model parameter of each class which determinate beta 

![](https://github.com/songge529/gibbs_program/raw/master/letter/beta.gif) - sampling from ![](https://github.com/songge529/gibbs_program/raw/master/letter/mu.gif) 

![](https://github.com/songge529/gibbs_program/raw/master/letter/H.gif) - which class the batch belongs to  

![](https://github.com/songge529/gibbs_program/raw/master/letter/H_pro.gif) - the probability the batch belongs to each class 

![](https://github.com/songge529/gibbs_program/raw/master/letter/v.gif), ![](https://github.com/songge529/gibbs_program/raw/master/letter/tau2.gif) - Hyperparameter

Update iteration order: ![](https://github.com/songge529/gibbs_program/raw/master/letter/Z.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/mu.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/beta.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/H_pro.gif),   ![](https://github.com/songge529/gibbs_program/raw/master/letter/H.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/v.gif),    ![](https://github.com/songge529/gibbs_program/raw/master/letter/tau2.gif)  

## The Algorithm Step : 

### step 1:
Get the Bayesian posterior probability derivation-- the Algorithm basis 

### step 2:

Initialize the generated simulation data 

### step 3:

Define a Gibbs update iterator function in the class  


### step 4:

Select ![](https://github.com/songge529/gibbs_program/raw/master/letter/K.gif)  -class number based on ![](https://github.com/songge529/gibbs_program/raw/master/letter/BIC.gif)


#### In addition :

**gibbs_with_same_obs.py**  --- under each batch having same observation number condition 

**gibbs_with_different_obsnumber.py**  ---under each batch having different observation number condition
