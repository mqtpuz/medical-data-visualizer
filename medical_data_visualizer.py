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
    df_cat = df_cat.value_counts().reset_index().rename(columns={0: 'total'})
    df_cat['value'] = df_cat.value.astype('str')  # to match the example figure

    # 7

    # 8
    fig = sns.catplot(data=df_cat, kind='bar', x='variable',
                      y='total', hue='value', col='cardio',
                      errorbar=None, height=4, aspect=4 / 3)

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    conditions = [(df.height >= df.height.quantile(0.025)),
                  (df.height <= df.height.quantile(0.975)),
                  (df.weight >= df.weight.quantile(0.025)),
                  (df.weight <= df.weight.quantile(0.975))]
    bool_mask = conditions[0] & conditions[1] & conditions[2] & conditions[3]

    df_heat = df.loc[bool_mask]

    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(corr)).astype('bool')

    # 14
    fig, ax = plt.subplots(layout='tight')

    # 15
    sns.heatmap(corr, annot=True, linewidth=0.5, mask=mask, cmap='coolwarm', center=0, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig
