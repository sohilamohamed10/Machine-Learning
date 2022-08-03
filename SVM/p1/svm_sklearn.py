import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

def svm_accuracy(data,normalize="false"):
	acc=[0,0]
	for i in range(10):
		x_train, x_test, y_train, y_test = train_test_split(data[:,:-1],data[:,-1], test_size = 0.4)
		clf = svm.SVC(kernel='linear')
		#scaling from 0 to 1
		norm=MinMaxScaler()
		x_train_norm=norm.fit_transform(x_train)
		x_test_norm=norm.fit_transform(x_test)
		train_data=[x_train,x_train_norm]
		test_data=[x_test,x_test_norm]
		#classification and calculating accuracy
		for j in range(2):
			clf.fit(train_data[j], y_train)
			y_pred = clf.predict(test_data[j])
			acc[j]=acc[j]+metrics.accuracy_score(y_test, y_pred)

	avg_acc1=acc[0]/10
	avg_acc2=acc[1]/10
	return avg_acc1,avg_acc2
#load data
data= np.loadtxt("data.txt", dtype=str) 
avg_acc1,avg_acc2=svm_accuracy(data)
print("Avg Accuracy without data normalization",avg_acc1)
print("Avg Accuracy with data normalization",avg_acc2)

