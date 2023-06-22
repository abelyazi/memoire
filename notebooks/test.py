import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import pandas as pd

path = r'df_complete_data.pkl'

df = pd.read_pickle(path)
print("allo")


#check correctness of dataframe loaded
for i in range(68261):
  if len(df["recording"][i]) != 4096:
    print("problem sir")


new_df = df[['patient_id', 'murmur']]
new_df = new_df.drop_duplicates()

new_df = new_df.reset_index(drop=True)

nb_murmur_present = (new_df.loc[new_df['murmur'] == 1]).shape[0]
nb_murmur_absent = (new_df.loc[new_df['murmur'] == 0]).shape[0]
print(nb_murmur_present)
print(nb_murmur_absent)

list_id = new_df['patient_id']
list_label = new_df['murmur']

def get_balanced_dataset(X,y):
  ones = np.where(np.array(y)==1)
  zeros = np.where(np.array(y)==0)
  ones = ones[0]
  zeros = zeros[0]
  trunc = ones.shape[0] - zeros.shape[0]
  zeros = zeros[:trunc]
  new_X=[]
  new_y=[]
  for i in ones:
    new_X.append(X[i])
    new_y.append(y[i])
  for j in zeros:
    new_X.append(X[j])
    new_y.append(y[j])
  return new_X,new_y

list_id,list_label = get_balanced_dataset(list_id,list_label)

id_train, id_valtest, label_train, label_valtest = train_test_split(list_id, list_label , test_size=0.2, random_state=42)
id_val, id_test, label_val, label_test = train_test_split(id_valtest, label_valtest , test_size=0.5, random_state=42)

def select_rows(id_list,dframe):
  #for id in list_id:
  sub_df = dframe.loc[dframe['patient_id'].isin([int(id) for id in id_list])]
  return sub_df

df_train = select_rows(id_train,df)
df_val = select_rows(id_val,df)
df_test = select_rows(id_test,df)

(df_train.loc[df_train['murmur'] == 0]).shape[0]

X_train = np.vstack(df_train['recording'])
y_train = np.array(df_train['murmur'])

X_val = np.vstack(df_val['recording'])
y_val = np.array(df_val['murmur'])

X_test = np.vstack(df_test['recording'])
y_test = np.array(df_test['murmur'])

X_train = X_train.reshape(len(X_train),len(X_train[1]),1)
X_val = X_val.reshape(len(X_val),len(X_val[1]),1)
X_test = X_test.reshape(len(X_test),len(X_test[1]),1)

def make_model(input_shape):
    drop_rate = 0.2
    input_layer = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    #input_layer = tf.keras.layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[0], mask_zero=True)(input_layer)
    #lstm1 = tf.keras.layers.LSTM(128)(input_layer)
    #lstm1 = tf.keras.layers.Dropout(0.2)(lstm1)
    #lstm2 = tf.keras.layers.LSTM(64)(lstm1)
    #lstm2 = tf.keras.layers.Dropout(0.2)(lstm2)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same",activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    #conv1 = tf.keras.layers.ReLU()(conv1)
    x = tf.keras.layers.MaxPool1D(2,padding="same")(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)


    #pool1 = tf.keras.layers.MaxPool1D(pool_size=(3,), padding='same')(conv1)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same",activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #conv2 = tf.keras.layers.ReLU()(conv2)
    x = tf.keras.layers.MaxPool1D(2, padding="same")(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    #x = tf.keras.layers.Flatten()(x)
    """
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    """
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=X_train.shape[1:])

print(model.summary())


