import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, max_depth=100 ):
        self.max_depth = max_depth
        self.root = None

    def compute_entropy(self,y):
        m=len(y)
        entropy=0
        if m!=0:
            p1=len(y[y==1])/m
            if p1!=0 and p1!=1:
                entropy=-p1*np.log2(p1)-(1-p1)*np.log2(1-p1)
            else :
                entropy=0

        return entropy

    def split_dataset(self,X,node_indices,feature,threshold):
        left_indices=[]
        right_indices=[]
        for i in node_indices:
            if X[i,feature]<=threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)

        return left_indices , right_indices

    def compute_information_gain(self,X,y,node_indices,feature,threshold):
        left,right=self.split_dataset(X,node_indices,feature,threshold)

        X_node,y_node=X[node_indices],y[node_indices]
        X_left,y_left=X[left],y[left]
        X_right,y_right=X[right],y[right]

        w_left=len(y_left)/len(y_node)
        w_right=len(y_right)/len(y_node)
        information_gain=self.compute_entropy(y_node)-(w_left*self.compute_entropy(y_left)+w_right*self.compute_entropy(y_right))

        return information_gain

    def best_split(self,X,y,node_indices):
        gains=[]
        best_feature=-1
        max_gain=-1
        n=X.shape[1]
        best_threshold=None
        for i in range(n):
            thresholds=np.unique(X[:,i])
            for threshold in thresholds:
                gain=self.compute_information_gain(X,y,node_indices,i,threshold)
                gains.append(gain)
                if gain>max_gain:
                    max_gain=gain
                    best_feature=i
                    best_threshold= threshold
                    
        return best_feature,best_threshold

    def build_tree(self,X,y,node_indices,branch_name,max_depth,current_depth):

        if current_depth==max_depth or len(np.unique(y[node_indices]))==1:
            counter = Counter(y)
            most_common_value = counter.most_common(1)[0][0]
            return Node(value=most_common_value )
        
        feature,threshold=self.best_split(X,y,node_indices)
        left,right=self.split_dataset(X,node_indices,feature,threshold)

        left=self.build_tree(X,y,left,"left",max_depth,current_depth+1)
        right=self.build_tree(X,y,right,"right",max_depth,current_depth+1)
        return Node(feature,threshold, left, right)
     
    def fit(self, X, y): 
        root_indices=list(np.arange(0,len(y)))  
        self.root=self.build_tree(X,y,root_indices,"Root",self.max_depth,0)
        
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
            
    def accuracy(self,y_true, y_pred):
            accuracy = np.sum(y_true == y_pred) / len(y_true)
            return accuracy

    def predict(self, X):
            return np.array([self.traverse_tree(x, self.root) for x in X])



#--------------
file = open("cardio_train.csv")
data_set = np.loadtxt(file, delimiter=";",skiprows=1,dtype=int)
x=data_set[:,1:-1]
y=data_set[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x[:2000],y[:2000],test_size=0.1, random_state=42)
clf=DecisionTree(max_depth=10)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
acc = clf.accuracy(y_test, y_pred)
print(acc)
   