# Challenge

##Questions: 

#### Q1. Regular find function for list
Describe (with pseudocode or real code) how to find an element by value in an unsorted list (i.e. define the "find" function for a list, with a list and a value as inputs). The function should return the index of the element found. What should you return if the element is not in the list?
> For simplicity, you can assume a list of integers.

## Attempt

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

## Attempt

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

## The code for Q1 and Q2 can be found in 'Q1.py'.

#### Q3. Machine learning for energy consumption
The `train_data.csv` contains energy consumption data of some store for some given days in the past where the store was working at an acceptable level of energy consumption (every day in the train_data can be considered good data, i.e. the store was working at an acceptable level).

The date, energy and enthalpy variables are given. A relationship between enthalpy (average daily outside air enthalpy) and energy (the total store energy consumption) exists (enthalpy is the independent variable, and energy is the dependent variable).

Create a machine learning model that would determine days where the store is not working optimally. Use the model to determine which days in the `test_data.csv` were at an acceptable level of energy consumption, and which days were not (energy consumption is too high).

Explain the method chosen, and why it was chosen. If you made any assumptions, describe them also. You can supply graphs and other visual aids to help explain your solution. Provide all the code as well.

For convenience, `data.csv` contains all data (both the train and the test data) and can be used if needed.

## Attempt

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

Here we find that the noise has a range of  <img src="https://render.githubusercontent.com/render/math?math=\pm 3 \sigma"> deviation from the best fit.

This allows us to set boundaries as to what is an acceptable energy level range for a given enthalpy value. Using our training (good data) we know that anything above or below that range is considered as unacceptable data. If we had training data that contained some bad data as well, we could have used a machine learning algorithm to train the data to recognize what is good and bad and set its own boundaries. (Classification problem).

- We now take our prediction model from Part 1 and apply it on the test_data given by 'test_data.csv' to recover new predictions for the given enthalpy.
- We then proceed to find all the dates when the energy values were within the defined range as shown in figure 'test_data.png' and below. The range was defined from the gaussian noise generated with standard deviation that we retrieved in our training set. All energy levels within the min and the max are deemed as acceptable.

![alt text](https://github.com/alexandrekhoury/Challenge/blob/main/test_data.png)

Figure 'good.png' shows the data that is between both the min and max of our fit and is shown below.

![alt text](https://github.com/alexandrekhoury/Challenge/blob/main/good.png)


#### On the plots we see that the energy and enthalpy values were renormalized. This was done in order to facilitate the machine learning algorithm. We can easily convert them back to their initial values since we kept track of the scaling transformation that was applied.

## The code for this procedure is found in 'Q3.py'.


#### Bonus Question: Produce the merge_ranges function.

Produce, in the language of your choice (Python strongly suggested), the `merge_ranges` function.

> This challenge is to produce the code to merge multiple ranges of values if the ranges intersect.

> The function should take a list of `DateRange`, and produces another list of `DateRange` which is the minimum representation of the input.

> A `DateRange` object is simply an object with a `start`, an `end`, with `start <= end`, where `start` and `end` are `date` objects. You can implement this object however you feel is most appropriate for the task.

Example:
Assume `date1 < date2 < date3 < date4 < date5 < date6`, then

+ `merge_ranges([DateRange(date1, date3), DateRange(date2, date4)]) == [DateRange(date1, date4)]`

+ `merge_ranges([DateRange(date1, date3), DateRange(date2, date5), DateRange(date4, date6)]) == [DateRange(date1, date6)]`

+ `merge_ranges([DateRange(date1, date2), DateRange(date3, date5), DateRange(date4, date6)]) == [DateRange(date1, date2), DateRange(date3, date6)]`

## Attempt

In my attempt I make the assumption that the first value entered in DateRange is smaller than the second value. (ex: for DateRange(a,b), a should be smaller than b) . 
I also make the assumption that dates are integers. If I had more time, I would code a function that turns the dates from strings to integers and then perform my analysis with the integers. 

## The code can be found in 'bonus.py'.

