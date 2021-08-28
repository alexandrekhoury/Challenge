
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
    
    
    #RESCALING 
    predictions_scaled=predictions*energy_scale.reshape(-1,1)
    test_predictions=test_predictions*test_energy_scale.reshape(-1,1)
    
    
    print("Days that had an acceptable level of energy consumption are:"+str(good_dates))
    print("Days that had an unacceptable level of energy consumption are:"+str(bad_dates))
                   
