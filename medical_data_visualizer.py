import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv', sep=',', index_col='id')

# 2
bmi = df.weight / (df.height / 100) ** 2
df['overweight'] = (bmi > 25).astype('int8')

# 3
for col in ['cholesterol', 'gluc']:
    df[col] = (df[col] > 1).astype('int8')


# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.value_counts().reset_index().rename(columns={0: 'total'}).sort_values(['variable'])
    df_cat['value'] = df_cat.value.astype('str')

    # 7
    fgrid = sns.catplot(df_cat, 
                      kind='bar', 
                      x='variable', 
                      y='total', 
                      hue='value', 
                      col='cardio', 
                      errorbar=None, 
                      height=4, 
                      aspect=4/3)

    # 8
    fig = fgrid.figure
    
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    cond = [(df.ap_lo <= df.ap_hi),
            (df.height >= df.height.quantile(0.025)),
            (df.height <= df.height.quantile(0.975)),
            (df.weight >= df.weight.quantile(0.025)),
            (df.weight <= df.weight.quantile(0.975))
            ]
    bool_mask = cond[0] & cond[1] & cond[2] & cond[3] & cond[4]
    
    df_heat = df.loc[bool_mask]
    df_heat.reset_index(inplace=True)

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr)).astype('bool')

    # 14
    fig, ax = plt.subplots(figsize=(9, 6), layout='tight')

    # 15
    sns.heatmap(corr, 
                annot=True, 
                linewidth=0.5, 
                mask=mask,  
                center=0,
                fmt='.1f',
                cbar_kws={'shrink': 0.8},
                ax=ax
               )

    # 16
    fig.savefig('heatmap.png')
    return fig
