#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import librosa

audio_dataset_path='UrbanSound8K/'
metadata=pd.read_csv('UrbanSound8K/UrbanSound8K.csv')
metadata.head()


# In[2]:


def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features
    


# In[3]:


import numpy as np
from tqdm import tqdm

extracted_features=[]
c=1
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"])).replace("\\","/")
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])
    c+=1


# In[4]:


### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[5]:


### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[6]:


X.shape


# In[7]:


y


# In[8]:


### Label Encoding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[9]:


y


# In[10]:


### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[11]:


X_train


# In[12]:


y


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


y_train.shape


# In[16]:


y_test.shape


# ### Model Creation

# In[17]:


import tensorflow as tf


# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics


# In[19]:


### No of classes
num_labels=y.shape[1]


# In[20]:


model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[21]:


model.summary()


# In[22]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[25]:


## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[26]:


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


# In[28]:


X_test[1]


# In[81]:


pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
pred


# ### Testing Some Test Audio Data
# 
# Steps
# - Preprocess the new audio data
# - predict the classes
# - Invere transform your Predicted Label

# In[86]:


filename="UrbanSound8K/dog_bark2.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=model.predict(mfccs_scaled_features)
predicted_label=np.argmax(predicted_label,axis=1)
predicted_label = np.array(predicted_label)
predicted_label = predicted_label.flatten()
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label) 
prediction_class


# In[ ]:




