
#Data preprocessing
#Fit in needed modeules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"C:/Users/crazy/Desktop")


#Read the data
df = pd.read_csv("high_diamond_ranked_10min.csv")
df.describe()



#There's no missing value in the dateset.
df.isnull().sum()

#Define y & x's for the dataset(df)
df_output = df[:]["blueWins"]
df_input_clean = df.drop(["gameId", "blueWins"], axis = 1)

#Reveal mean and std.
print("mean and std of all features：")
df_stats = df_input_clean.describe().loc[['mean', 'std']]
df_stats

#copy
df_input = df_input_clean.copy()

#list out of all columns
print("Columns : ")
column_names = [i for i in df_input.columns]
for i in column_names:
    print(i)

#regularization
#StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScale
#We choose MaxAbs. for the reason that it holds the highest value of prediction vlaue and explaination power in PCA.
from sklearn.preprocessing import MaxAbsScaler 

Z2 = df_input

scaler = MaxAbsScaler()
scaler.fit(Z2)
Z = scaler.transform(Z2)

df_input.iloc[:,:] = Z2


#Reveal mean and std.
print("mean and std of all features：")
df_stats_Z = df_input.describe().loc[['mean', 'std']]
df_stats_Z







#End of the data preprocessing. And we have our regularized data : (data_input or Z).
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


x = np.asarray(df_input, dtype="float32")
y = to_categorical(df_output, dtype="int")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)


y_train_F= np.asarray([y_train[i][0] for i in range(len(y_train))])
y_test_F= np.asarray([y_test[i][0] for i in range(len(y_test))])



#LinearRegression
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


lm = LinearRegression(fit_intercept=True, n_jobs=None)
lm.fit(x_train, y_train)

#original regression
y_pred_lm = lm.predict(x_test)
R2_lm2 = str(round(metrics.r2_score(y_test, y_pred_lm)*100, 4))
AR2_lm2 = str(round((1 - (1-metrics.r2_score(y_test, y_pred_lm))*(len(y_test) - 1)/(len(y_test) - x_train.shape[1] - 1))*100, 4))
print("R^2 : " + R2_lm2)
print("Adj. R^2 : " + AR2_lm2)

#new recommendation of lm.
lm_pred = np.argmax(y_pred_lm,axis=1)
acc_lm = str(round(accuracy_score(y_test_F, lm_pred)*100,4))
print("acc. of lm : " + acc_lm)

#Of course that linear regression has poor predictive power in the binary output scenario.







#LogisticRegression
from sklearn.linear_model import LogisticRegression
lrm = LogisticRegression(max_iter=10000)
lrm.fit(x_train, y_train_F)

pred_lm = lrm.predict(x_test)
acc_lrm2 = str(round(accuracy_score(y_test_F, pred_lm)*100,4))
print("the accuracy of lm prediction is : " + acc_lrm2)




#DecisionTree
from sklearn.tree import DecisionTreeRegressor
dct = DecisionTreeRegressor()
dct.fit(x_train, y_train)

y_pred_dct = dct.predict(x_test)
acc_dct2 = str(round(metrics.accuracy_score(y_test, y_pred_dct)*100,4))
print("acc of decision tree model : " + acc_dct2)





#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rfm = RandomForestRegressor()
rfm.fit(x_train, y_train)

y_pred_rf = rfm.predict(x_test)
y_pred_rf = np.argmax(y_pred_rf,axis=1)

acc_rfm2 = str(round(metrics.accuracy_score(y_test_F, y_pred_rf)*100,4))
print("acc. of rf model : " + acc_rfm2)



#SVM
from sklearn import svm
svmr = svm.SVR()
svmr.fit(x_train, y_train_F)
y_pred_svm = svmr.predict(x_test)
acc_svmr2 = str(round(metrics.r2_score(y_test_F, y_pred_svm)*100,4))
print("acc. of svm model : " + acc_svmr2)

#sum_up matrix
no_reel = pd.DataFrame({"no_process" : { "linear reg." : acc_lm, "logistic reg." : acc_lrm2, "decision tree" : acc_dct2, "random forest" : acc_rfm2, "svm" : acc_svmr2} }) 
print(no_reel)




#Redundant of 14 columns for they are merely negative value of the opponent's team.
elim_col = ["blueCSPerMin", "redCSPerMin", "blueGoldPerMin", "redGoldPerMin", "redKills", "redDeaths", "redFirstBlood", "blueGoldDiff", "blueExperienceDiff"]
for i in elim_col:
  if i in df_input:
    df_input = df_input.drop(columns = i)
  else:
    continue
#Eliminate of low correlation variables by setting at 0.3
corr_list = df_input[df_input.columns[:]].apply(lambda x: x.corr(df_output))
cols = []
for col in corr_list.index:
    if (corr_list[col]>0.3 or corr_list[col]<-0.3):
        cols.append(col)

print("Seleccted columns : ")
df_input = df_input_clean[cols]
df_input.columns

#regularization
#StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScale
#We choose MaxAbs. for the reason that it holds the highest value of prediction vlaue and explaination power in PCA.
from sklearn.preprocessing import MaxAbsScaler 

Z = df_input

scaler = MaxAbsScaler()
scaler.fit(Z)
Z = scaler.transform(Z)

df_input.iloc[:,:] = Z


#Reveal mean and std.
print("mean and std of all features：")
df_stats_Z = df_input.describe().loc[['mean', 'std']]
df_stats_Z

#End of the data preprocessing. And we have our regularized data : (data_input or Z).

#Decomposition by method of PCA which is a way to extract the important information by dimensions reduction
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
L = pca.fit_transform(df_input)


#Explain percentage
pca_explained = PCA()
pca_explained.fit(df_input)
p_pca = np.round(pca_explained.explained_variance_ratio_, 2)
print("The 2 components explain percentage : " + str(round(p_pca[0]+p_pca[1],4)))

#Visual data & preparation
df_vis = pd.DataFrame(data = L, columns = ['pc1', 'pc2'])
df_vis = pd.concat([df_vis, df['blueWins']], axis = 1)
df_vis

#By showing the result of PCA, we can roughly explain the key factors contributing the winning rate for the blue team.

#Meanings of tho components
pcs = np.array(pca.components_) # (n_comp, n_features)
df_pc = pd.DataFrame(pcs, columns = df_input.columns[:])

#For Jupyter Notebook
df_pc.index = [f"comp. {c}" for c in["one", "two"]]
df_pc.style.background_gradient(cmap = "bwr_r", axis = None)

#For pc_1, the latter two column have high contribution to the data. while the second one (pc_2) has highest value in the first two columns.

#Once again.
#List out of all columns
column_names = [i for i in df_input.columns]
column_names

#We can conclude that :
#blueKills and blueDeaths are one key factor.
#redGoldDiff and redExperienceDiff (red minus blue) are another key factor.
#To sum up, we can conculde that the above factors do profoundly affect the winning rate for the blue team.





#Model train
x = np.asarray(df_input, dtype="float32")
y = to_categorical(df_output, dtype="int")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)


y_train_F= np.asarray([y_train[i][0] for i in range(len(y_train))])
y_test_F= np.asarray([y_test[i][0] for i in range(len(y_test))])


"""
"""

#LinearRegression

lm = LinearRegression(fit_intercept=True, n_jobs=None)
lm.fit(x_train, y_train)

#original regression
y_pred_lm = lm.predict(x_test)
R2_lm2 = str(round(metrics.r2_score(y_test, y_pred_lm)*100, 4))
AR2_lm2 = str(round((1 - (1-metrics.r2_score(y_test, y_pred_lm))*(len(y_test) - 1)/(len(y_test) - x_train.shape[1] - 1))*100, 4))
print("R^2 : " + R2_lm2)
print("Adj. R^2 : " + AR2_lm2)

#new recommendation of lm.
lm_pred = np.argmax(y_pred_lm,axis=1)
acc_lm = str(round(accuracy_score(y_test_F, lm_pred)*100,4))
print("acc. of lm : " + acc_lm)

#Of course that linear regression has poor predictive power in the binary output scenario.







#LogisticRegression
lrm = LogisticRegression(max_iter=10000)
lrm.fit(x_train, y_train_F)

pred_lm = lrm.predict(x_test)
acc_lrm2 = str(round(accuracy_score(y_test_F, pred_lm)*100,4))
print("the accuracy of lm prediction is : " + acc_lrm2)




#DecisionTree
dct = DecisionTreeRegressor()
dct.fit(x_train, y_train)

y_pred_dct = dct.predict(x_test)
acc_dct2 = str(round(metrics.accuracy_score(y_test, y_pred_dct)*100,4))
print("acc of decision tree model : " + acc_dct2)





#RandomForestRegressor
rfm = RandomForestRegressor()
rfm.fit(x_train, y_train)

y_pred_rf = rfm.predict(x_test)
y_pred_rf = np.argmax(y_pred_rf,axis=1)

acc_rfm2 = str(round(metrics.accuracy_score(y_test_F, y_pred_rf)*100,4))
print("acc. of rf model : " + acc_rfm2)



#SVM
svmr = svm.SVR()
svmr.fit(x_train, y_train_F)
y_pred_svm = svmr.predict(x_test)
acc_svmr2 = str(round(metrics.r2_score(y_test_F, y_pred_svm)*100,4))
print("acc. of svm model : " + acc_svmr2)

#

#sum_up matrix
with_reel = pd.DataFrame({"with_process" : { "linear reg." : acc_lm, "logistic reg." : acc_lrm2, "decision tree" : acc_dct2, "random forest" : acc_rfm2, "svm" : acc_svmr2} }) 
print(with_reel)

final_result = pd.concat([no_reel, with_reel], axis = 1)
final_result
"""
"""


#nueral network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# NN Model stacking.
model_nn = Sequential()
model_nn.add(Dense(10, input_dim = len(df_input.columns)))
model_nn.add(Dense(128, activation='relu'))
model_nn.add(Dense(128, activation='relu'))
model_nn.add(Dense(128, activation='relu'))
model_nn.add(Dense(1))
model_nn.compile(loss = "mse", optimizer = "Adam", metrics = ["accuracy"], run_eagerly = True)

model_nn.summary()

# NN Model training.
# This dataset is a small one, which may somehow occur overfitting problem.

nn_result = model_nn.fit(x_train, y_train_F, batch_size = 320, epochs = 20, validation_split = 0.10)




fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(nn_result.history["accuracy"], color = "g", label = "accuracy")
ax2.plot(nn_result.history["loss"], color = "b", label = "Loss")

ax1.set_ylabel("accuracy", color = "g")
ax2.set_ylabel("loss", color = "b")

plt.title("Model Accuracy")
plt.xlabel("epoch")
fig.legend()
plt.show()

print("Brief view on the result of training.")







# Examine the process of NN_model on loss and R2 score.
plt.figure(figsize = (10, 8))
plt.subplot(211)  

plt.plot(nn_result.history["loss"])
plt.plot(nn_result.history["val_loss"])
plt.title("Model Loss Value")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"], loc = "upper left")


plt.subplot(212)  

plt.plot(nn_result.history["accuracy"])
plt.plot(nn_result.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Training", "Validation"], loc = "upper left")
plt.show()

# Accuracy of well-trained NN Model.
scores = model_nn.evaluate(x_test, y_test_F)
       
print(final_result)
print("acc. of nn is : " + str(round(scores[1],4)*100))       
                  