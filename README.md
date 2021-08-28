# Challenge

##Questions: 

#### Q1. Regular find function for list
Describe (with pseudocode or real code) how to find an element by value in an unsorted list (i.e. define the "find" function for a list, with a list and a value as inputs). The function should return the index of the element found. What should you return if the element is not in the list?
> For simplicity, you can assume a list of integers.

##Attempt

There are many ways to go about this question. Here is code in python for one of the possible ways: 
1. Find it using the basic methods: 
ex code: 

```ruby
List = [1,2,3,4] 
def find(x,List):
    count=0
    for i in List :
        if x==i: 
            return count
        count+=1
    return print("The value was not found.")
    
print(find(3,List))
```

Here x should be an integer and List should be a list of integers.

#### Q2. Optimised find function for sorted list
How would optimise the "find" function of Q1 if the input list is a sorted list?

##Attempt

For a sorted array, one can use the Binary search algorithm which can be coded in python as follows: 
ex code: 

```
import math
List = [1,28,34,48,72,102] 
def find_sorted(x,List):
    L=0
    R=len(List)-1	
    while (L<=R):
        i= math.floor((L+R)/2)
        if List[i]<x: 
            L+=1
        elif List[i]>x:
            R=i-1
        else:
            return i       
    return print("The value was not found.")
    
print(find_sorted(72,List))
```

Here we have an implementation of the Binary search algorithm that takes the middle value of a list and sees if our target value is greater of less than our middle value and repeatedly iterates until the value index is found.

#### Q3. Machine learning for energy consumption
The `train_data.csv` contains energy consumption data of some store for some given days in the past where the store was working at an acceptable level of energy consumption (every day in the train_data can be considered good data, i.e. the store was working at an acceptable level).

The date, energy and enthalpy variables are given. A relationship between enthalpy (average daily outside air enthalpy) and energy (the total store energy consumption) exists (enthalpy is the independent variable, and energy is the dependent variable).

Create a machine learning model that would determine days where the store is not working optimally. Use the model to determine which days in the `test_data.csv` were at an acceptable level of energy consumption, and which days were not (energy consumption is too high).

Explain the method chosen, and why it was chosen. If you made any assumptions, describe them also. You can supply graphs and other visual aids to help explain your solution. Provide all the code as well.

For convenience, `data.csv` contains all data (both the train and the test data) and can be used if needed.

##Attempt

##### Part 1 (Relationship between energy and enthalpy)

- We start by using keras API in tensorflow. At first glance this seems like we are tackling a non-linear regression problem (after having plotted the data).
- The data that was used for this part was from a training data set that had all acceptable values (no values are rejected). 
- We use the Sequential class to create a multilayer neural network. Here we use 2 hidden layers with 16 neurons (found by trial and error) and one output layer. 
- We use elu activation functions that work better with our model (they are smoother than relu) and we used a linear activation function for our output layer.
- We then set the optimizer to be Adam which is an algorithm that is stochastic gradient descent method (which is typical for these types of problems).
- We then set our loss to be computed with the mean_squared_error (typical for regression problems). 
- Finally we fit the model and obtain our predictions on the relation between energy and enthalpy that is plotted below and in 'fit.png'.
![alt text](https://github.com/alexandrekhoury/Challenge/blob/main/fit.png)

##### Part 2 (Classification of test_data)

There are a couple ways to approach this problem. We can train our data using unsupervised machine learning. Here I describe another approach that I took to solve this problem.

- Using the data for the fit (predictions) from Part 1, we find the residuals of the data from the fit. 
- Then we calculate the standard deviation of the residuals. This will let us generate some gaussian noise with the same standard deviation.
- We apply the gaussian noise that was generated on our fit and retrieve the max and minimum values of the fit. 
- We now see that all data points are well located between our max and min. (To be more precise in the future, we can divide our energy in a certain amount of parts (that seem to have a similar range of noise) and find the standard deviation of these parts of the residuals and apply it to the parts of the fit. This would help us to not overestimate the size that is acceptable). A figure is presented in 'range.png' and below.

![alt text](https://github.com/alexandrekhoury/Challenge/blob/main/range.png)

This allows us to set boundaries as to what is an acceptable energy level range for a given enthalpy value. Using our training (good data) we know that anything above or below that range is considered as unacceptable data. If we had training data that contained some bad data as well, we could have used a machine learning algorithm to train the data to recognize what is good and bad and set its own boundaries. (Classification problem).

- We now take our prediction model from Part 1 and apply it on the test_data given by 'test_data.csv' to recover new predictions for the given enthalpy.
- We then proceed to find all the dates when the energy values were within the defined range as shown in figure 'test_data.png' and below. The range was defined from the gaussian noise generated with standard deviation that we retrieved in our training set. All energy levels within the min and the max are deemed as acceptable.

![alt text](https://github.com/alexandrekhoury/Challenge/blob/main/test_data.png)

Figure 'good.png' shows the data that is between both the min and max of our fit and is shown below.

![alt text](https://github.com/alexandrekhoury/Challenge/blob/main/good.png)


#### On the plots we see that the energy and enthalpy values were renormalized. This was done in order to facilitate the machine learning algorithm. We can easily convert them back to their initial values since we kept track of the scaling transformation that was applied.

The code for this procedure is found in 'Q3.py' and below. 

```ruby
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_data(filename):
    with open(filename) as f:
        #skipping lines that are comments
        lines = (line for line in f if not line.startswith('#'))
        #unpacking energy and enthalpy values 
        energy,enthalpy = np.loadtxt(
            lines,
            dtype='float,float',
            delimiter=',',
            skiprows=1, #skip first line with header
            usecols=(1,2),
            unpack=True
            )
        #unpacking the date values as strings
        date = np.loadtxt(filename,
                          delimiter=',',
                          dtype='str',
                          skiprows=1,
        
                          usecols=(0))
    return energy,enthalpy,date

def scaling(variable):
    scale=variable/((variable-variable.min())/(variable.max()-variable.min()))
    variable=((variable-variable.min())/(variable.max()-variable.min()))
    return scale, variable

def model(energy,enthalpy):
    model=tf.keras.Sequential(
    [
     tf.keras.layers.Dense(16,
                           input_dim=1,
                           activation='elu'),
     tf.keras.layers.Dense(16,activation='elu'),
     tf.keras.layers.Dense(1)
     ])

    optimizer= tf.keras.optimizers.Adam(learning_rate = 1e-3)
    
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse']
                  )
    
    model.fit(
        x=enthalpy, y=energy, batch_size=5, epochs=50
        )
    
    predictions = model.predict(enthalpy)
        
    return predictions,model

def find_noise(std,size):
    noise = np.random.normal(0,std,size)
    noise_max=noise.max()
    noise_min=noise.min()
    return noise,noise_max,noise_min

if __name__=='__main__':
       
    #unpacking the data and scaling
    train_data='train_data.csv'
    energy_raw,enthalpy_raw,date=load_data(train_data)
    energy_scale,energy=scaling(energy_raw)
    enthalpy_scale,enthalpy=scaling(enthalpy_raw)   

    #plotting data
    plt.figure()
    plt.plot(enthalpy,energy,'o',label='raw data')
    plt.xlabel("Renormalized enthalpy")
    plt.ylabel("Renormalized energy")
    
    #reshaping for machine learning algorithm
    energy=energy.reshape(-1,1)
    enthalpy=enthalpy.reshape(-1,1)
    
    #finding best model using machine learning algorithm (linear regression means squared error)
    predictions,model=model(energy,enthalpy)
    

    #plotting fit
    plt.plot(enthalpy,predictions,'o',label='trained fit')
    plt.xlabel("Renormalized enthalpy")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig('fit.png')


    #adding our gaussian noise to our fit
    residuals=energy-predictions
    std=np.std(residuals)
    noise,noise_max,noise_min=find_noise(std,len(predictions))
        
    #plotting fit with noise
    plt.figure()
    plt.plot(enthalpy,predictions+noise_max,label='Max prediction range')
    plt.plot(enthalpy,predictions+noise_min,label='Min prediction range')
    plt.plot(enthalpy,energy,'o',label='raw data')
    plt.xlabel("Renormalized enthalpy")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig('range.png')
    
    #=================================
    #repeating steps for the test data
    #=================================
    
    #unpacking data and scaling
    test_data='test_data.csv'
    test_energy_raw,test_enthalpy_raw,test_date=load_data(test_data)
    test_energy_scale,test_energy=scaling(test_energy_raw)
    test_enthalpy_scale,test_enthalpy=scaling(test_enthalpy_raw)   
     
    #reshape array for machine learning algorithm
    test_energy=test_energy.reshape(-1,1)
    test_enthalpy=test_enthalpy.reshape(-1,1)
    
    #applying algorithm and finding the noise using same standard deviation as in our trained data
    test_predictions=model.predict(test_enthalpy)
    test_noise,test_noise_max,test_noise_min=find_noise(std,len(test_predictions))
    
    #plotting test_data
    plt.figure()
    plt.plot(test_enthalpy,test_predictions+noise_max,label='Max prediction range')
    plt.plot(test_enthalpy,test_predictions+noise_min,label='Min prediction range')
    plt.plot(test_enthalpy,test_energy,'o',label='raw data')
    plt.legend()
    plt.savefig('test_data.png')
    
    #want our values to be between min and max of noise
    logic1=test_energy<(test_predictions+test_noise_max) 
    logic2=test_energy>(test_predictions+test_noise_min)
    good=np.ravel(logic1*logic2)
    
    #finding the dates were data is acceptable
    good_dates=test_date[good]
    bad_dates=test_date[np.invert(good)]
    
    #plotting data from within range only
    plt.figure()
    plt.plot(test_enthalpy,test_predictions+noise_max,label='Max prediction range')
    plt.plot(test_enthalpy,test_predictions+noise_min,label='Min prediction range')
    plt.plot(test_enthalpy[good],test_energy[good],'o',label='raw data')
    plt.legend()
    plt.savefig('good.png')
    
    #RESCALING 
    predictions_scaled=predictions*energy_scale.reshape(-1,1)
    test_predictions=test_predictions*test_energy_scale.reshape(-1,1)
    
    
    print("Days that had an acceptable level of energy consumption are:"+str(good_dates))
    print("Days that had an unacceptable level of energy consumption are:"+str(bad_dates))
```
