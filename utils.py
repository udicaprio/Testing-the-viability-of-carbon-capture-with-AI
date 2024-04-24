import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
class ModelWrapper:
  def __init__(self, model, Xscaler, Yscaler):
    self.model = model
    self.Xscaler = Xscaler
    self.Yscaler = Yscaler

  def predict(self, x):
    x = np.atleast_2d(x)
    xsc = self.Xscaler.transform(x)
    ysc = self.model(xsc).numpy()
    y = self.Yscaler.inverse_transform(ysc)
    return y
  
  def __cost_function_KGa(self, x):
    return -self.predict(x)[:,0]

  def maximize_KGa(self, bounds, 
                   CO2pp_index, CO2pp_value, 
                   CO2alpha_index, CO2alpha_value):
    bounds = np.asarray(bounds)
    bounds[CO2pp_index] = [CO2pp_value, CO2pp_value]
    bounds[CO2alpha_index] = [CO2alpha_value, CO2alpha_value]
    return differential_evolution(self.__cost_function_KGa, bounds, seed = 0)

def parity_plot(Ytrain, Ytest, Ytrain_pred, Ytest_pred):

  s = np.vstack((Ytrain, Ytest, Ytrain_pred, Ytest_pred))
  bounds = np.hstack((s.min(axis=0).reshape(-1,1), 
                      s.max(axis=0).reshape(-1,1)))
  
  fig, ax = plt.subplots(1, Ytrain.shape[1])
  for k in range(Ytrain.shape[1]):
    ax[k].plot(Ytest[:,k], Ytest_pred[:,k], '.', label = 'Test set')
    ax[k].plot(Ytrain[:,k], Ytrain_pred[:,k], '.', alpha = 0.2, label = 'Train set')
    ax[k].plot(bounds[k], bounds[k], '--', color = 'red', label = 'Ideal prediction')
    ax[k].set_xlabel('Experimental value')
    ax[k].set_ylabel('Predicted value')
  ax[-1].legend()
  plt.tight_layout
  plt.show()


def visualize_solution(best_x, df):
  fig, ax = plt.subplots(3, 2)
  ax[0, 0].set_title(df.columns[0])
  ax[0, 0].plot(df.iloc[:,0], df.iloc[:,0], '.')
  ax[0, 0].plot(best_x[0], best_x[0], '.', color = 'red')

  ax[1, 0].set_title(df.columns[1])
  ax[1, 0].plot(df.iloc[:,1], df.iloc[:,1], '.')
  ax[1, 0].plot(best_x[1], best_x[1], '.', color = 'red')

  ax[2, 0].set_title(df.columns[2])
  ax[2, 0].plot(df.iloc[:,2], df.iloc[:,2], '.')
  ax[2, 0].plot(best_x[2], best_x[2], '.', color = 'red')

  ax[0, 1].set_title(df.columns[3])
  ax[0, 1].plot(df.iloc[:,3], df.iloc[:,3], '.')
  ax[0, 1].plot(best_x[3], best_x[3], '.', color = 'red')

  ax[1, 1].set_title(df.columns[4])
  ax[1, 1].plot(df.iloc[:,4], df.iloc[:,4], '.')
  ax[1, 1].plot(best_x[4], best_x[4], '.', color = 'red')

  ax[2, 1].set_title(df.columns[5])
  ax[2, 1].plot(df.iloc[:,5], df.iloc[:,5], '.')
  ax[2, 1].plot(best_x[5], best_x[5], '.', color = 'red')

  plt.tight_layout()
  plt.show()