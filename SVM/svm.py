import numpy as np
from sklearn.model_selection import train_test_split
from operator import itemgetter
import matplotlib.pyplot as plt
class SVM():
	def __init__(self,lamda=0.0001,iters=1000,learning_rate=0.01):
		self.features=None
		self.targets=None
		self.w=None
		self.b=None
		self.lamda=lamda
		self.iters=iters
		self.lr=learning_rate
		
	def fit(self,x,y):
		self.features=x
		self.targets=y
		self.w=np.zeros(self.features.shape[1])
		self.b=0
		self.classes=[]
		self.costs=[]
		self.plot_idx1=[]   #for ploting function
		self.plot_idx2=[]
		#one to all classification (multiclass->binary class)
		for t in range(len(self.targets)):
			if self.targets[t]=="Iris-virginica":
				self.plot_idx1.append(t)
				self.classes.append(-1)
			else:
				self.classes.append(1)
				self.plot_idx2.append(t)
		# Gradient descent concept
		for i in range(self.iters):
			j=0
			dw=np.zeros((x.shape[0],x.shape[1]))
			db=np.zeros((x.shape[0],1))
			for k in range (len(x)):
				margin=self.classes[k]*(np.dot(self.features[k],self.w)+self.b)
				j+=self.cost_fn(margin)
				
				if margin < 1:
					dw[k]=2*self.lamda*self.w-self.classes[k]*self.features[k]
					db[k]=-self.classes[k]
				else:
					dw[k]=2*self.lamda*self.w
					db[k]=0
			self.costs.append(j+self.lamda*0.5*np.dot(self.w,self.w))
			self.w=self.w-self.lr*np.sum(dw,axis=0)
			self.b=self.b-self.lr*np.sum(db,axis=0)
			
			#condition for termination(cost function becomes almost constant)
			if(i>0):
				if (abs(self.costs[i]-self.costs[i-1])<0.001):
					break
		
	def cost_fn(self,margin):
		#j=lamda∑(w^2/2)+∑max(0,1-yi(w.xi+b))
		j=np.maximum(0,1-margin)
		return j

	def plot(self,x):

		first_class=itemgetter(*self.plot_idx1)(x.tolist())
		second_class=itemgetter(*self.plot_idx2)(x.tolist())
		plt.scatter([x[0] for x in first_class],[x[1] for x in first_class] , s = 50, c = '#1f77b4', alpha = 0.8)
		plt.scatter([x[0] for x in second_class],[x[1] for x in second_class], s = 50, c = '#ff7f0e', alpha = 0.8)

		ax = plt.gca()
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		xx = np.linspace(xlim[0], xlim[1], 30)
		yy = np.linspace(ylim[0], ylim[1], 30)
		YY, XX = np.meshgrid(yy, xx)
		xy = np.vstack([XX.ravel(), YY.ravel()]).T
		Z = (np.dot(xy,self.w)+self.b).reshape(XX.shape)
		ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])
 
		plt.show()


	def predict(self,x):
		predictions=[]
		p=np.dot(x,self.w)+self.b
		for i in range(p.shape[0]):
			if p[i]<0:
				predictions.append(-1)
			else:
				predictions.append(1)
		return predictions
	
	def calc_acc(self,pred,y_true):
		targets=[]
		count=0
		for t in y_true:
			if t=="Iris-virginica":
				targets.append(-1)
			else:
				targets.append(1)
		for s in range(len(y_true)):
			if pred[s]==targets[s]:
				count=count+1	
		return (count/len(pred))*100



#---------------------------
#example
file = open("Iris.csv")
data_set = np.loadtxt(file, delimiter=",",skiprows=1,dtype=str)
data_set=data_set[data_set[:,-1]!="Iris-versicolor"]
x=data_set[:,1:3]
x=x.astype(np.float)
y=data_set[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.4,random_state=1)
clf=SVM()
clf.fit(X_train,y_train)
p=clf.predict(X_test)
acc=clf.calc_acc(p,y_test)
clf.plot(X_train)
print(acc)



