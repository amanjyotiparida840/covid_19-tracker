import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffle=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffle[:test_set_size]
    train_indices=shuffle[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


if __name__ == '__main__':
  df = pd.read_csv('democorona.csv')
  train,test=data_split(df,0.2)
  X_train=train[['Fever','BodyPain','Age','RunnyNose',
                 'DifficultBreadth']].to_numpy()
  X_test=test[['Fever','BodyPain','Age','RunnyNose'
      ,'DifficultBreadth']].to_numpy()

  Y_train=train[['InfectionProbable']].to_numpy().reshape(1677,)
  Y_test=test[['InfectionProbable']].to_numpy().reshape(419,)

  cfl=LogisticRegression()
  cfl.fit(X_train,Y_train)
  inputfeatures=[100,1,99,1,0]
  infect=cfl.predict([inputfeatures])
  inf_prob = cfl.predict_proba([inputfeatures])[0][1]


  file= open('model.pkl','wb')
  pickle.dump(cfl,file)
  file.close()


