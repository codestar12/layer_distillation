#%%
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
with open('./logs/refactor_log.json', 'r') as f:
    log = json.load(f)
#%%
sns.lineplot(x=[layer['layer'] for layer in log['layer']],
             y=[layer['train_time'] for layer in log['layer']])

#%%
data_frame_list = []
for layer in log['layer']:
    data_frame_list.append((
        layer['layer'],
        layer['loss'],
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
plt.plot()
sns.lineplot(x='layer', y='acc', hue='stage', data=df)
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


