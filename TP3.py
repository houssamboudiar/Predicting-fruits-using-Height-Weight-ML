#------------------------- Etape 1 ---------------------------------

import matplotlib as plt
import pandas as pd 
pd.set_option('max_rows',5) #shows only 5 rows from the whole dataset
import scipy.stats as sps

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression # 

from adspy_shared_utilities import plot_class_regions_for_classifier_subplot # for the optional part !

#------------------------- Etape 2 ---------------------------------

#fuits csv read using pandas
FruitDataSet = pd.read_csv("fruit_data_with_colors.csv") # read the csv file using pandas then putting its values on FruitDataSet (which is just a Pandas DataFrame data type) 

FruitDataSet.head()  # shows the first 5 rows // without it it would print "row1""row2" ... "row(N-1)""row(N)"
print("-----------------------fruits table ------------------------------------")
print(FruitDataSet) # printing our DataFrame to the console
print(FruitDataSet.shape) # shape is one of the DataFrame attributes it shows the number of rows and columns
print("-----------------------END------------------------------------")

features = FruitDataSet.columns[-4:].tolist() # colums is also DataFrame attribute it returns a list 
# [-4:] means return only the last 4 elements of the list == ['mass', 'width', 'height', 'color_score']
# cause  "features = FruitDataSet.columns[:-4].tolist()" will return ['fruit_label', 'fruit_name', 'fruit_subtype']

print("DataSet Features : ")
print(features) # printing features :p

targetNames = ['apple','mandarin','orange','lemon']
print(targetNames)

print("-----------------------")
print("-----------------------")
print("-----------------------")



X = FruitDataSet[['height','width']] #return a dataframe with only the mentioned columns 
print(X)

Y = FruitDataSet[['fruit_label']] #return a dataframe with only the mentioned columns 
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

"""
applying Split funtion on the X , Y DataFrames
80% for the training x_train (80% of the X dataframe) y_train (80% of the Y dataframe)
20% for the test x_test (20% of the X dataframe) y_test (20% of the Y dataframe)
"""

print("\nX_train:\n")
print(x_train.head())
print(x_train.shape)

y_train = y_train == 1
y_train = y_train.astype(int)
""" astype(int) used in order to convert boolean array to integer array of ones and zeros which is more efficient 
you can use boolean arrays but it's deprecated and will be removed on the next updates you will also get that error
            DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
"""
print("\ny_train:\n")
print(y_train.head())
print(y_train.shape)

print("\nX_test:\n")
print(x_test.head())
print(x_test.shape)

y_test = y_test == 1
y_test = y_test.astype(int)
print("\ny_test:\n")
print(y_test.head())
print(y_test.shape)

print("-----------------------")
print("-----------------------")
print("-----------------------")
# make into a binary problem : apples vs everything else 
# as we can see on the csv file label = 1 refers to apple | if that value of the Y Dataframe is equal to 1 is true means its an apple // else its not an apple :p
# make into a binary problem : apples vs EVERYTHING ELSE LOL
#Y_apple = y_train == 1 # that returns a dataframe Y_apple contains our Dataset indexes from 1 - 58 and a boolean fruit_label column (true if the label = 1 = apple) (false if it's 2 , 3 or 4 = mandarin , orange or lemon )
#print(Y_apple.astype(int)) # printing that dataframe
"""              TO CLEAR OUT THAT PART EVEN THAT CODE WOULD WORK JUST FINE
                              Y = FruitDataSet[['fruit_name']]
                              Y_apple = Y == 'apple'
                              print(Y_apple)
"""
#------------------------- Etape 3 optionelle ---------------------------------
#fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))

#plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None,
#                                          None, 'Logistic regression \
#for binary classification\\nFruit dataset: Apple vs others',
#                                          subaxes)

#subaxes.set_xlabel('height')
#subaxes.set_ylabel('width')

# remarque on peut pas afficher dans l'ordre que le prof a mentionné, les variables n'etaient pas encore declarés (???)

#------------------------- Etape 4 ---------------------------------
clf = LogisticRegression() # instance of the logistic regression model

#clf.fit(X,Y_apple) 
# using dataframe as an input for fit() funtion is deprecated converting that data frame to 1D array using ravel() funtion would be more efficient
#print(Y_apple.values.ravel())
# that line of code would be :

#y_train = y_train == 1
#print(y_train.values)

clf.fit(x_train , y_train.values.ravel()) 


#------------------------- Etape 5 ---------------------------------
h = 6 
w = 8

print('A fruit with height {} and width {} is predicted to be: {}'.format(h, w, ['Everything else','apple'][clf.predict([[h, w]])[0]]))

h = 10
w = 7

print('A fruit with height {} and width {} is predicted to be: {}'.format(h, w, ['Everything else','apple'][clf.predict([[h, w]])[0]]))

# ------------------- étape 6 -----------------------

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(x_train,y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(x_test,y_test)))