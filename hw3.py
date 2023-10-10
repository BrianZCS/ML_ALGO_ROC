import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## take numpy array
## calculate the distance between one point to many points
def euclidean_d (point, points):
    return np.sqrt(np.sum((point-points)**2,axis=1))

def knn_y (train_data, test_data,k):
    y_pred = []
    for j in range(len(test_data)):
        distance = euclidean_d(np.array(test_data.iloc[j,:-1]),np.array(train_data.iloc[:,:-1]))
        neighbour = {"y_value": train_data.y,"distance":distance}
        neighbour = pd.DataFrame(neighbour, columns = ['y_value','distance'])
        neighbour = neighbour.sort_values(by = 'distance', kind='mergesort').iloc[:k,] 
        values, counts = np.unique(neighbour.y_value, return_counts=True)
        y_pred.append(int(values[counts == counts.max()][0]))
    return y_pred

def knn (train_data, test_data,k):
    y_pred = []
    for j in range(len(test_data)):
        distance = euclidean_d(np.array(test_data.iloc[j,:-1]),np.array(train_data.iloc[:,:-1]))
        neighbour = {"y_value": train_data.Prediction,"distance":distance}
        neighbour = pd.DataFrame(neighbour, columns = ['y_value','distance'])
        neighbour = neighbour.sort_values(by = 'distance', kind='mergesort').iloc[:k,] 
        values, counts = np.unique(neighbour.y_value, return_counts=True)
        y_pred.append(int(values[counts == counts.max()][0]))
    return y_pred
    
def knn_confidence (train_data, test_data,k):
    confidence_positive = []
    for j in range(len(test_data)):
        distance = euclidean_d(np.array(test_data.iloc[j,:-1]),np.array(train_data.iloc[:,:-1]))
        neighbour = {"y_value": train_data.Prediction,"distance":distance}
        neighbour = pd.DataFrame(neighbour, columns = ['y_value','distance'])
        neighbour = neighbour.sort_values(by = 'distance', kind='mergesort').iloc[:k,] 
        temp = neighbour.loc[neighbour.y_value==1,'y_value']
        confidence_positive.append(len(temp)/k)
    return confidence_positive

def cross_validation(k, data, model,model_parameter):
    accuracy = []
    precision = []
    recall = []
    for i in range(k):
        test_data = data.iloc[i*len(data)//k:(i+1)*len(data)//k,:]
        training_data_front = data.iloc[0:i*len(data)//k,:]
        training_data_back = data.iloc[(i+1)*len(data)//k:,:]
        training_data = pd.concat([training_data_front,training_data_back],ignore_index=True, sort=False)
        if(model==gradient_descent):
            theta = model(5000, training_data, model_parameter)
            values = logistic_reg(theta,test_data)
            for i in range(len(values)):
                if values[i] >= 0.5:
                    values[i] = 1
                else:
                    values[i] = 0
            test_data.insert(test_data.shape[1],"model_prediction",values)
        else:
            test_data.insert(test_data.shape[1],"model_prediction",model(training_data,test_data,model_parameter))
        accuracy.append(len(test_data[test_data['Prediction']==test_data['model_prediction']])/len(test_data))
        precision.append(len(test_data[(test_data['Prediction']==1)&(test_data['model_prediction']==1)])/(len(test_data[(test_data['Prediction']==1) & (test_data['model_prediction']==1)])+len(test_data[(test_data['Prediction']==0) & (test_data['model_prediction']==1)])))
        recall.append(len(test_data[(test_data['Prediction']==1)&(test_data['model_prediction']==1)])/(len(test_data[(test_data['Prediction']==1) & (test_data['model_prediction']==1)])+len(test_data[(test_data['Prediction']==1) & (test_data['model_prediction']==0)])))
    return (accuracy, precision, recall)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logistic_reg(theta, data):
    temp = np.array(data.iloc[:,:-1]).dot(theta.T)
    return sigmoid(temp)

def gradient_descent(epoch, data, lamda):
    theta = np.repeat(0,data.shape[1]-1)
    for i in range(epoch):
        step = (logistic_reg(theta, data)-data.Prediction).dot(data.iloc[:,:-1])/len(data)
        theta = theta - lamda * step
    return theta

def confidence_level(data,model,model_parameter):
    i = 4
    k = 5
    test_data = data.iloc[i*len(data)//k:(i+1)*len(data)//k,:]
    training_data_front = data.iloc[0:i*len(data)//k,:]
    training_data_back = data.iloc[(i+1)*len(data)//k:,:]
    training_data = pd.concat([training_data_front,training_data_back],ignore_index=True, sort=False)
    if(model==gradient_descent):
        theta = model(5000, training_data, model_parameter)
        values = logistic_reg(theta,test_data)
        test_data.insert(test_data.shape[1],"confidence_positive",values)
    else:
        test_data.insert(test_data.shape[1],"confidence_positive",model(training_data,test_data,model_parameter))
    test_data = test_data.sort_values(by =['confidence_positive','Prediction'], kind='mergesort',ascending=[False,True])
    return test_data

def roc_curve(data):
    threshoulds = [{'FPR':0, 'TPR':0}]
    num_neg = len(data[data['Prediction']==0])
    num_positive = len(data[data['Prediction']==1])
    TP = 0
    FP = 0
    last_TP = 0
    for i in range(0,len(data)):
        if((i > 0) and (data.iloc[i,-1]!=data.iloc[i-1,-1])
           and (data.iloc[i,-2]==0) and (TP>last_TP)):
            FPR = FP/num_neg
            TPR = TP/num_positive
            threshoulds.append({'FPR':FPR, 'TPR':TPR})
            last_TP = TP
        if data.iloc[i,-2] == 1:
            TP += 1 
        else:
            FP += 1
    FPR = FP/num_neg
    TPR = TP/num_positive
    threshoulds.append({'FPR':FPR, 'TPR':TPR})
    return threshoulds

#1.5 Create ROC Curve
x = [0,0,1/4,2/4,1]
y = [0,2/6,4/6,1,1]
plt.scatter(x,y,s=100,color = 'r',edgecolors= "black")
plt.plot(x,y,'r-',linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()         

#2.1
f = open("D2z.txt", "r")
x1 = []
x2 = []
y = []
for line in f:
    temp = line.split()
    x1.append(temp[0])
    x2.append(temp[1])
    y.append(temp[2])
train_data = pd.DataFrame(list(zip(x1, x2, y)), columns =['x1', 'x2','y'], dtype = float)
x1 = np.arange(-2, 2, 0.1)
x2 = np.arange(-2, 2, 0.1)
l1 = len(x1)
l2 = len(x2)
x1 = np.repeat(x1, l2)
x2 = np.tile(x2, l1)
test_data = pd.DataFrame(columns=['x1', 'x2', 'y'], dtype = float)
test_data['x1']=x1
test_data['x2']=x2
plt.scatter(train_data.loc[train_data.y==0,'x1'], train_data.loc[train_data.y==0,'x2'],edgecolors= "black",color = 'white',label="class 0")
plt.scatter(train_data.loc[train_data.y==1,'x1'], train_data.loc[train_data.y==1,'x2'],color= "black",marker="+",label="class 1")
test_data['y'] = knn_y(train_data,test_data,1)
plt.scatter(test_data.loc[test_data.y==0,'x1'], test_data.loc[test_data.y==0,'x2'],color = 'blue',label="class 0",alpha = 0.5,s=10)
plt.scatter(test_data.loc[test_data.y==1,'x1'], test_data.loc[test_data.y==1,'x2'],color= "red",label="class 1", alpha = 0.5, s=10)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Predictions of 1NN on a 2D Grid")
plt.legend()
plt.show()

## 2.2
emails = pd.read_csv('emails.csv',index_col=0)
accuracy, precision, recall = cross_validation(5,emails, knn,1)
print("Accuracy:", accuracy, "Pression:",precision, "Recall:",recall)

## 2.3
accuracy, precision, recall = cross_validation(5,emails, gradient_descent,0.0005)
print("Accuracy:", accuracy, "Pression:",precision, "Recall:",recall)

## 2.4
accuracy_k = [] 
emails = pd.read_csv('emails.csv',index_col=0)
accuracy, precision, recall = cross_validation(5,emails, knn,1)
accuracy_k.append({'k':1, 'accuracy':np.mean(accuracy)})
accuracy, precision, recall = cross_validation(5,emails, knn,3)
accuracy_k.append({'k':3, 'accuracy':np.mean(accuracy)})
accuracy, precision, recall = cross_validation(5,emails, knn,5)
accuracy_k.append({'k':5, 'accuracy':np.mean(accuracy)})
accuracy, precision, recall = cross_validation(5,emails, knn,7)
accuracy_k.append({'k':7, 'accuracy':np.mean(accuracy)})
accuracy, precision, recall = cross_validation(5,emails, knn,10)
accuracy_k.append({'k':10, 'accuracy':np.mean(accuracy)})
df = pd.DataFrame(accuracy_k)
plt.plot(df.k,df.accuracy,'-o',c='steelblue')
plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.title('kNN 5-Fold Cross Validation')
plt.grid()
plt.show()

## 2.5
confidence_logreg = confidence_level(emails, gradient_descent,0.0005)
confidence_knn = confidence_level(emails, knn_confidence, 5)
temp = roc_curve(confidence_knn)
df = pd.DataFrame(temp)
plt.plot(df.FPR,df.TPR,'-',c='steelblue',label = "KNeighborsClassifier(AUC = "+str(round(np.trapz(df.TPR,df.FPR),2))+")")
temp = roc_curve(confidence_logreg)
df = pd.DataFrame(temp)
plt.plot(df.FPR,df.TPR,'-',c='orange',label = "LogisticRegression(AUC = "+str(round(np.trapz(df.TPR,df.FPR),2))+")")
plt.xlabel('False Positive Rate (Positive label: 1)')
plt.ylabel('True Positive Rate (Positive label: 1)')
plt.title('ROC Curve')
plt.grid()
plt.legend()
plt.show()