
#Assume `date1 < date2 < date3 < date4 < date5 < date6`

import numpy as np


#defining the object
class DateRange:
  def __init__(self, start, end):
    self.start = start
    self.end = end
    
    #this function makes sure that if two objects have same attributes (start)
    #and end, they will be equal to each other
  def __eq__(self, other): 
      if not isinstance(other, DateRange):
          return NotImplemented

      return self.start == other.start and self.end == other.end



def merge_ranges(date_list):
    
    #used for itiration
    count=0
    dates=np.zeros([len(date_list),2])
    
    #unpacking the date list into an array
    for date in date_list:
        dates[count]=np.array([date.start,date.end])
        count+=1
    
    #sort dates with earliest start date
    dates=dates[np.argsort(dates[:,0])]
    
    #defining array that will store the new dates
    new_dates=[0,0]
    
    #defining the variables that will get us out of the while loop
    shape=1
    new_shape=0

    #we have a while loop since we analyse each element of the list with the next one once
    #then we repeat the process until it is optimized
    #it is optimized when we can't merge the dates anymore and hence the shapes of the old array
    #is the same as the new array
    while new_shape!=shape:

        #initializing variables
        skip=-100000000
        shape=dates.shape[0]
    
        for i in range(0,dates.shape[0]):
            
            #skip values that are between the interval that is being merged
            if i<=skip :
                continue
            
            start=dates[i,0]
            
            #logistics for array length
            if i==dates.shape[0]-1:
                new_dates=np.vstack([new_dates,[start,dates[i,1]]])
            else:
                
                for j in range(i+1,dates.shape[0]):
                    
                    #trying to see if first end date is bigger than second start date
                    if dates[i,1]<dates[j,0]:
                        
                        end= dates[j-1,1]
                        new_dates=np.vstack([new_dates,[start,end]])
                        break
                    #trying to see if first end date is bigger than second end date
                    if dates[i,1] < dates[j,1]:
                        
                        end= dates[j,1]
                        new_dates=np.vstack([new_dates,[start,end]])
                        skip=j
                        break
                    # adding this in case the whole date range object is encompassed in another one
                    if dates[i,1]>=dates[j,0] and dates[i,1] >= dates[j,1]:
                        end=dates[i,1]
                        new_dates=np.vstack([new_dates,[start,end]])
                        skip=j
                        break
                    

        #delete values where start is 0 in array
        new_dates=np.delete(new_dates, 0, axis=0)
        #swapping
        dates=new_dates
        new_dates=[0,0]
        new_shape=dates.shape[0]
    
    #converting our array of merged dates into objects
    count=0
    new_date_list=[]
    for date in dates:
        
        new_date_list=np.append(new_date_list,DateRange((date[0]),date[1]))

    return list(new_date_list)


if __name__=='__main__':
       
    date_list=[DateRange(2,5),DateRange(1,3),DateRange(6,8),DateRange(7,14),DateRange(15,57)]
        
    new_date_list=merge_ranges(date_list)
    
    date1=1
    date2=2
    date3=3
    date4=4
    date5=5
    date6=6
    
    print(merge_ranges([DateRange(date1, date3), DateRange(date2, date4)]) == [DateRange(date1, date4)])
    print(merge_ranges([DateRange(date1, date3), DateRange(date2, date5), DateRange(date4, date6)]) == [DateRange(date1, date6)])
    print(merge_ranges([DateRange(date1, date3), DateRange(date2, date5), DateRange(date4, date6)]) == [DateRange(date1, date6)])