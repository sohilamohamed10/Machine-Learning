import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
class regression():
    def __init__(self,data):
       self.features=None
       self.y=None
       self.weights=np.zeros(len(data.columns)-1)
       self.bias=0
       self.cost_fns=[]
    
    def _predict (self,x):
        #y=w.x+b
        return np.dot(x,self.weights)+self.bias
        
    def cost_function(self,y_pred): 
        #loss=∑(y-h)^2
        cost_fn=np.sum(np.power((self.y-y_pred),2))
        return cost_fn

    def gradient_descent(self,y_pred,lrate):
        #dw , db
        gradient=-2*np.dot((self.y-y_pred),self.features)     
        gradient_bias=-2*np.sum(self.y-y_pred)
        self.weights=self.weights-lrate*gradient
        self.bias=self.bias-lrate*gradient_bias
        y_pred_new=self._predict(self.features)
        return y_pred_new

    def fit(self,x,y,lrate=0.01): 
        self.features=x
        self.y=y
        #normalization if data is multivariate
        if(self.features.shape[1]>1):
            self.features=(self.features-self.features.min(axis=0))/(self.features.max(axis=0)-self.features.min(axis=0))

        y_pred=self._predict(self.features)
        cost=self.cost_function(y_pred)
        self.cost_fns.append(cost)
        c=1
        while(True):
            y_pred=self.gradient_descent(y_pred,lrate)
            self.cost_fns.append(self.cost_function(y_pred))
            #if cost fn is increasing reduce lr
            if (self.cost_fns[c]>self.cost_fns[c-1]):
                lrate=lrate*0.2
                #terminate when cost fn is almost constant
            if(abs(self.cost_fns[c]-self.cost_fns[c-1])<0.0001):
                print(self.cost_fns[c])
                break
            c=c+1

    def predict(self,X_test):
        #normalization of test data in multivariate data
        if(X_test.shape[1]>1):
            X_test=(X_test-X_test.min(axis=0))/(X_test.max(axis=0)-X_test.min(axis=0))
        return np.dot(X_test,self.weights)+self.bias

    def evaluate (self,y,y_true):
        error=[]
        #mean absolute error mertic
        for i in range(len(y)) :
            diff=y[i]-y_true[i]
            error.append(abs((y[i] - y_true[i])/y_true[i]))
        accuracy=(np.sum(error)/len(y))
        return accuracy*100
 
 #-------------------------------
 #example                
data= pd.read_table('multivariateData.dat' ,sep=",",header=None)
x=data.iloc[:,:-1].to_numpy()
y=data.iloc[:,-1].to_numpy()
r=regression(data)
X_train, X_test, y_train,y_test = train_test_split(x, y,test_size=0.2)
r.fit(X_train,y_train)
predictions=r.predict(X_test)
accuracy=r.evaluate(predictions,y_test)
print(accuracy)
