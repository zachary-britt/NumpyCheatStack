import pandas as pd
import numpy as np


def read_with_pandas(filename):
    #df = pd.read_csv(filename, header=None)
    df = pd.read_csv(filename)
    print(df.head())
    col0 = df.index
    row0 = df.loc[0]
    print(row0)
    row0 = df.iloc[1]
    print(row0)
    return df


def interaction(row):
    return row['med_2011'] + int(row['med_2014']>5)

# apply function to rows elementwise
def dfapply():
    df = pd.read_csv('ass2/data/rent.csv')
    #apply across rows
    df['inter'] = df.apply(interaction, axis=1)
    print(df.head())


def to_numpy(df):
    #.values turns dataframe into numpy array
    X = df.values
    Y = df.as_matrix()
    Z = np.array(df)

    row0 = X[0] #(np is row first, column second)
    col0 = df[0] #pandas only allows column indexing


def merges(df):
    #making a new DataFrame
    col_list = list(df.columns)
    col_list.remove('Year')
    df2= pd.DataFrame()
    df2['Year'] = df.Year
    df2['Total'] = df[col_list].sum(axis=1)
    #print(df2.head())

    df=df.merge(df2, on='Year')
    #df.merge(right, how='inner', on=None, left_on=None, right_on=None,
    #   left_index=False, right_index=False, sort=False,
    #   suffixes=('_x', '_y'), copy=True, indicator=False)
    print(df.head())


def groups(df):

    gb = df.groupby(['State','City'])
    print(type(gb))
    #gb is a groupby object until we run an aggregation
    gbm=gb.mean()
    print(type(gbm))
    gbms=gbm.sort_values('med_2014', ascending=False)
    print(gbms.head())

if __name__ == "__main__":
    filename='ass2/data/rent.csv'
