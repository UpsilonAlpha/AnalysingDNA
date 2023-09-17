import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

#370 Tumor Samples
#50 Normal Tissue Samples
#420 Total Samples

#Reading data
dnameth = pd.read_csv("G9_liver_dna-meth.csv")
rnaseq = pd.read_csv("G9_liver_gene-expr.csv")

#Pre-processing data
dnameth.dropna()
rnaseq.dropna()
dnameth.iloc[:, 1].unique()
dnameth = dnameth.rename(columns = {'Primary Tumor':'0'})
rnaseq.iloc[:, 1].unique()
rnaseq.rename(columns = {'Primary Tumor':'0'})

#Defining dependant variable
y = dnameth['Label'].values

#Encoding categorical data
Labelencoder = LabelEncoder()
Y = Labelencoder.fit_transform(y) # Primary Tumor = 0, Solid Tissue Normal = 1

# Define the independent variables to drop the Label and Unnamed: 0
X = dnameth.drop(labels = ['Label','Unnamed: 0'], axis = 1)
feature_names = np.array(X.columns)

#Normalisation
scaler = StandardScaler()
scaler.fit(X)
X_scale = scaler.transform(X)
X_log2 = np.log2(X)
X_log2 = scaler.fit_transform(X_log2)


X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size = 0.3, random_state = 42)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# geneMean = rows.mean()

# print(geneMean)

# fig = px.scatter(geneMean, log_y=True)

# fig.show()


#Save 15 normal tissue for testing
#Save 111 tumor samples for testing

# mean = 0
# for column in rnaseq:
#     print(column)
#     for ind in rnaseq.index:
#         if rnaseq["Label"][ind] == "Solid Tissue Normal":
#             mean = mean + column[ind]
#     mean = (mean/50)
#     print(mean)

