import numpy as np
import math

#------------------------------------------------------------------

def rel_error(predictions_tensor, targets_array):
 
  predictions = np.array(predictions_tensor).T
  targets = targets_array.T
  n = len(predictions[0]) # number of datapoints to avg
  error = []
  
  for i in range(len(predictions)):
    soma = 0
    for j in range(n):
      if(targets[i][j]!=0):
        soma += np.abs((predictions[i][j]-targets[i][j])/targets[i][j])
        #erro = np.abs((predictions[i][j]-targets[i][j])/targets[i][j])
        #print("(i,j) = (%d, %d): %f" % (i,j, erro ))
    error.append(soma/n)
  return error


#-------------------------------------------------------------------

def densitie_fraction(Y_array):
  list=[]
  for _ in Y_array.T:
    list.append(_/2.56e22)
  return np.array(list).T

#-------------------------------------
# def standarization(Y_array):
#   def std_transform(array):
#     return (array-np.mean(array))/np.std(array)

#   list=[]
#   for _ in Y_array.T:
#     list.append(std_transform(_))
#   return np.array(list).T

class Standardize():
  mean = None
  std = None

  def standardization(self, array):
    self.mean = np.mean(array)
    self.std = np.std(array)
    return (array-self.mean)/self.std

#-------------------------------------

def log_transform(array):
  list=[]
  for _ in array.T:
    max = np.max(_)
    min = np.min(_)
    print(_)
    result = [math.log(item/np.sqrt(max*min), max/min) for item in _]
    list.append(result)
  return  np.array(list).T # (value, base)