
# coding: utf-8

# # Guide for Exploratory data analysis and model building for iris species identification

# ### Required modules for a kick start with the data exploration :- DESCRIPTIVE ANALYSIS

# ##### Basic guadiance on the below Libraries
#       Numpy & Pandas : Help in structuring & exploring the data
#       Seaborn & Matplotlib : Helps in pictorical presentation of the data 
#       %matplotlib inline : Just for Jupyter users 

# In[1]:


import numpy as numpy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ##### Call out the dataset : You can find the dataset in my respository

# In[2]:


iris = pd.read_csv(r"C:\Users\megha\Desktop\DATASET\IRIS\tableconvert_csv.csv")


# ##### First glance of iris dataset

# In[3]:


df = iris.head().append(iris.tail()) #To understand about the attributes of the dataset


# In[4]:


df #Top 5 and below 5 


# In[5]:


iris.describe() #Analysing the data by the means of statistics (Mean, median, Std. , percentiles & counts)


# In[6]:


iris.info() #Heads up for any missing values in the dataset or duplicacy of the columns 


# In[7]:


iris.shape #Quick check on no. of Rows & Columns


# In[8]:


iris.columns # Target class 


# In[9]:


iris.species.value_counts() #Examine the target class to see if we have evenly distributed data or not  


# ### DESCRIPTIVE ANALYSIS

# In[10]:


sns.pairplot(iris,markers='s',hue='species') # Setosa can be linearly separable 


# In[11]:


correlation = iris.corr()
print(correlation)


# In[12]:


# Pearson correlation of features 
plt.figure(figsize=(14,8))
sns.heatmap(correlation,annot=True,cmap=sns.color_palette('coolwarm'),linewidths=.5,fmt='.2g')


# In[13]:


sns.violinplot(y='species',x='sepal_length',data=iris)


# In[14]:


sns.violinplot(y='species',x='sepal_width',data=iris)


# In[15]:


sns.violinplot(y='species',x='petal_length',data=iris)


# In[16]:


sns.violinplot(y='species',x='petal_width',data=iris)


# ### Required modules for model building | Classification Machine Learing |

# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# ##### Prep for modeling 

# In[18]:


x = iris.iloc[:,:-1]
y = iris.iloc [:,-1]


# In[19]:


x.head()


# In[20]:


y.head()


# In[21]:


model = LogisticRegression()


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 6)


# In[23]:


model.fit(x_train,y_train)


# In[24]:


#test the model 
predictions = model.predict(x_test)


# In[25]:


print(predictions)

print(y_test)


# In[26]:


# Model prediction evaluation 
print(classification_report(y_test,predictions))


# In[27]:


print(accuracy_score(y_test,predictions))


# ##### pip install  --user --upgrade scikit-learn== 0.22.2  (Bonus : Upgrade Scikit learn) 
