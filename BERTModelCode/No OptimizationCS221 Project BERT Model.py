# this is with no optimization for progress report

#!/usr/bin/env python
# coding: utf-8

# ## **Patient Message Categorization with BERT Model**
# 
# The project concerns patient message categorization (multi-class text classification) based on their messages by using pre-trained Distilbert model. Based on given text as an input, we have predicted which medical personal should respond to the message. In our analysis we have used a Huggingface (transformers) library as well.
# 
# This project is inspired by: https://github.com/aniass/Product-Categorization-NLP
# 

# In[1]:


#!pip install transformers


# **Importing the required libraries**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras import utils as np_utils
from keras.utils.np_utils import to_categorical

import transformers
from transformers import AutoTokenizer,TFDistilBertModel, DistilBertConfig
from transformers import TFAutoModel

import warnings
warnings.filterwarnings("ignore")


# In[3]:


print(tf.__version__)
print(keras.__version__)


# In[4]:


# cd "../"


# We load previous cleaned up dataset.

# In[5]:


# CHANGE: for testing faster, read only the first n row data with param: nrows=n
df = pd.read_csv('patientMessagesDataset5-18.csv', header=0,index_col=0, nrows=1000)
# CHANGE: remove respondent data with < 9000 messages (44 -> 6 classes)
# respondent_types_to_remove = []
# for val, cnt in df['respondent_type'].value_counts().iteritems():
#     if cnt < 9000:
#         respondent_types_to_remove.append(val)
# df = df[df['respondent_type'].isin(respondent_types_to_remove) == False]
df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


#types of categories
df['respondent_type'].value_counts()


# ### **Data preparation**
# 
# Spliting the data into train and test sets:

# In[9]:


X = df['msg_txt']
y = df['respondent_type']


# In[10]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# In[11]:


encoder = LabelEncoder()
encoder.fit(y_train)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

num_classes = np.max(y_train) + 1

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# ### **DistilBERT model**
# 
# **DistilBERT**is a distilled version of BERT: smaller,faster, cheaper and lighter. It is a small, fast, cheap and light Transformer model trained by distilling BERT base. Because the BERT model has large size, it is difficult fot it to put it into production. Sometimes we want to use these model when we need a less weight yet efficient model. That's when we can use Distil-BERT model. It is  a smaller general-purpose language representation model, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts. It has 40% less parameters than bert-base-uncased and runs 60% faster. It also has 97% of BERT’s performance while being trained on half of the parameters of BERT. 
# 
# In our task we have a small dataset and this model can be a good choice to try for us. 
# 
# In the first step we have to make tokenization on our dataset. Tokenization will allow us to feed batches of sequences into the model at the same time.
# 
# To do the tokenization of our datasets we have to choose a pre-trained model. We load the Distilbert model `(distilbert-base-uncased) `from the Huggingface Transformers library.

# In[12]:


# Creating tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


# Now we have to load Distilbert model. In the Transformers library is avaliable Distilbert model and we use the `TFAutoModel` model (distilbert-base-uncased). 
# 
# Note: the red message is expected.

# In[13]:


bert = TFAutoModel.from_pretrained('distilbert-base-uncased')


# In[14]:


for layer in bert.layers:
      layer.trainable = True


# The function which allows to encode our dataset with tokenizer. We have decided on a maximum sentence length is 100 (maxlen).

# In[15]:


# Tokenization of the data
def text_encode(text, tokenizer, max_len=100):
    tokens = text.apply(lambda x: tokenizer(x,return_tensors='tf', 
                                            truncation=True,
                                            padding='max_length',
                                            max_length=max_len, 
                                            add_special_tokens=True))
    input_ids= []
    attention_mask=[]
    for item in tokens:
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
    input_ids, attention_mask=np.squeeze(input_ids), np.squeeze(attention_mask)

    return [input_ids,attention_mask]


# Based on this encodings for our training and testing datasets are generated as follows:

# In[16]:


X_train_input_ids, X_train_attention_mask = text_encode(X_train, tokenizer, max_len=100)
X_test_input_ids, X_test_attention_mask = text_encode(X_test, tokenizer, max_len=100)


# ### **Build the model**
# 
# We create a Distilbert model with pretrained weights and then we add two Dense layers with Dropout layer.

# In[17]:


# model creation
def build_model(bert_model, maxlen=100):
   input_ids = tf.keras.Input(shape=(maxlen,),dtype=tf.int32, name='input_ids')
   attention_mask = tf.keras.Input(shape=(maxlen,),dtype=tf.int32, name='attention_mask')

   sequence_output = bert_model(input_ids,attention_mask=attention_mask)
   output = sequence_output[0][:,0,:]
   output = tf.keras.layers.Dense(32,activation='relu')(output)
   output = tf.keras.layers.Dropout(0.2)(output)
# CHANGE: 5 to y_train.shape[1]
   output = tf.keras.layers.Dense(y_train.shape[1],activation='softmax')(output)

   model = tf.keras.models.Model(inputs = [input_ids,attention_mask], outputs = [output])
   model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

   return model


# In[18]:


model = build_model(bert, maxlen=100)


# We train the model for 10 epoch:

# In[ ]:


history = model.fit(
    [X_train_input_ids, X_train_attention_mask],
    y_train,
    batch_size=32,
    validation_data=([X_test_input_ids, X_test_attention_mask], y_test),
    epochs=2
)


# Visualization of training:

# In[ ]:


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


# In[ ]:


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# Predictions on a test set:

# In[ ]:


loss, accuracy = model.evaluate([X_test_input_ids, X_test_attention_mask], y_test)
print('Test accuracy :', accuracy)


# ### **Summary**
# 
# For our analysis we have used a pretrained Distilbert model to resolve our  text classification problem. After trained model we achieved an accuracy on the test set equal to 93 % and it is a similar result in comparison to previous  models that we have used. We also tested a several models by adding layers and increase numbers of epochs but we do not achaived a better accuracy. 
# 
