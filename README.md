# gibbs_program
# Gibbs Sampling Program

The algorithm is based on  **Bayesian Collaborative Model** aimed to fix the problem that the regression parameter is larger than the observation number and which could get the Estimation of parameters Mu ,Beta and H Simultaneously.Because the data is real Pharmaceutical production data, different  medicines have different batches. Each batch is response variable, each batch has different production records. Therefore, we assumed that some different batch maybe belong to same class with some attributes.   

## Parameter Description: 

$$Z$$ - Variable selection with 0 or 1 $Z_ij$ = 1 means $\mu_ij$ != 0 

![](https://github.com/songge529/gibbs_program/raw/master/letter/mu.gif) - model parameter of each class which determinate beta 

\beta - sampling from $\mu$ 

$H$ - which class the batch belongs to  

$H_pro$ - the probability the batch belongs to each class 

$v$, $tau^2$ - Hyperparameter

Update iteration order: $Z$,$\mu$,$\beta$,$H_pro$,$H$,$v$,$tau^2$  

## The Algorithm Step : 

### step 1:
Get the Bayesian posterior probability derivation-- the Algorithm basis 

### step 2:

Initialize the generated simulation data 

### step 3:

Define a Gibbs update iterator function in the class  


### step 4:

Select $K$-class number based on $BIC$


#### In addition :

**gibbs_with_same_obs.py**  --- under each batch having same observation number condition 

**gibbs_with_different_obsnumber.py**  ---under each batch having different observation number condition
