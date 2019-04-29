#%%
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#%%
files = os.listdir('./logs/vgg16_cifar10/')

#%%
with open('./logs/refactor_log.json', 'r') as f:
    log = json.load(f)
#%%
sns.lineplot(x=[layer['layer'] for layer in log['layer']],
             y=[layer['train_time'] for layer in log['layer']])

#%%
data_frame_list = []
for file in files:
    with open('./logs/vgg16_cifar10/' + file) as f:
        log = json.load(f)
    for layer in log['layer']:
        data_frame_list.append((
            layer['layer'],
            layer['loss'],
            log['model_epocss'],
            log['layer_epochs'],
            layer['acc'],
            layer['model_loss'],
            layer['train_time'],
            log['original_acc'],
            log['original_loss'],
            log['train_time'],
            'layer'
        ))
    for layer in log['layer']:
        data_frame_list.append((
            layer['layer'],
            layer['loss'],
            log['model_epocss'],
            log['layer_epochs'],
            layer['fine_tune_acc'],
            layer['fine_tune_model_loss'],
            layer['train_time'],
            log['original_acc'],
            log['original_loss'],
            log['train_time'],
            'fine_tune'
        ))

#%%
columns = ['layer',
           'layer_loss',
           'fine_tune_epochs',
           'layer_epochs',
           'acc',
           'model_loss',
           'layer_train_time',
           'original_acc',
           'original_loss',
           'model_train_time',
           'stage']

#%%
df = pd.DataFrame(data_frame_list, columns=columns)

#%%
top_df = df[df['layer_epochs'] == 4]
top_df = top_df[top_df['fine_tune_epochs'] == 8]
#%%
plt.plot()
sns.lineplot(x='layer', y='acc', hue='stage', data=top_df)
plt.hlines(y=log['original_acc'], 
           xmin=1, 
           xmax=12, 
           colors=['red'],
           linestyles='dashed')
plt.show()

#%%
plt.plot()
sns.lineplot(x='layer', y='model_loss', hue='stage', data=top_df)
plt.hlines(y=log['original_acc'], 
           xmin=1, 
           xmax=12, 
           colors=['red'],
           linestyles='dashed')
plt.show()



#%%
plt.plot()
sns.lineplot(x='layer', y='layer_train_time',  data=df)
plt.show()

#%%
new_df = df[df['layer'] == 12]
new_df = new_df[new_df['stage'] == 'fine_tune']
#%%
plt.plot()
sns.lineplot(x='fine_tune_epochs', y='acc', hue='layer_epochs', data=new_df )
plt.hlines(y=log['original_acc'], 
           xmin=1, 
           xmax=16, 
           colors=['red'],
           linestyles='dashed')
plt.show()




