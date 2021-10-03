import glob
import os
import random
from difflib import SequenceMatcher
import time

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # to remove some warnings

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from lineup_builder import Lineup

# Helper Functions
def get_weekly_data(week, year):
    """ get player data for designated week """
    file_path = f"./csv's/{year}/year-{year}-week-{week}-DK-player_data.csv"
    df = pd.read_csv(file_path)
    return df

def get_ytd_season_data(year, current_week):
    """ get data for current season up to most recent week """
    df = get_weekly_data(1,year)
    for week in range(2,current_week+1):
        try:
            df = df.append(get_weekly_data(week, year), ignore_index=True)
        except:
            print("No data for week: "+str(week))
    df = df.drop(['Unnamed: 0', 'Year'], axis=1)
    return df

def get_season_data(year, drop_year=True):
    """ get entire season of data """
    df = get_weekly_data(1,year)
    for week in range(2,17):
        try:
            df = df.append(get_weekly_data(week, year), ignore_index=True)
        except:
            print("No data for week: "+str(week))
    if drop_year:
        df = df.drop(['Unnamed: 0', 'Year'], axis=1)
    else:
        df = df.drop(['Unnamed: 0'], axis=1)
    return df

def scale_features(sc_salary, sc_points, sc_pts_ald, X_train, X_test, first_time=False):
    """ scales data for training """
    if first_time:
        X_train['DK salary'] = sc_salary.fit_transform(X_train['DK salary'].values.reshape(-1,1))
#         X_train['Oppt_pts_allowed_lw'] = sc_pts_ald.fit_transform(X_train['Oppt_pts_allowed_lw'].values.reshape(-1,1))
    X_test['DK salary'] = sc_salary.transform(X_test['DK salary'].values.reshape(-1,1))
#     X_test['Oppt_pts_allowed_lw'] = sc_pts_ald.transform(X_test['Oppt_pts_allowed_lw'].values.reshape(-1,1))
    return X_train, X_test

def unscale_features(sc_salary, sc_points, sc_pts_ald, X_train, X_test):
    """ used to change features back so that human readable information can be used to assess
    lineups and player information and performance"""
    X_train['DK salary'] = sc_salary.inverse_transform(X_train['DK salary'].values.reshape(-1,1))
#     X_train['Oppt_pts_allowed_lw'] = sc_pts_ald.inverse_transform(X_train['Oppt_pts_allowed_lw'].values.reshape(-1,1))
    X_test['DK salary'] = sc_salary.inverse_transform(X_test['DK salary'].values.reshape(-1,1))
#     X_test['avg_points'] = sc_points.inverse_transform(X_test['avg_points'].values.reshape(-1,1))
#     X_test['Oppt_pts_allowed_lw'] = sc_pts_ald.inverse_transform(X_test['Oppt_pts_allowed_lw'].values.reshape(-1,1))
    return X_train, X_test

def handle_nulls(df):
    # players that have nulls for any of the columns are 
    # extremely likely to be under performing or going into a bye.
    # the one caveat is that some are possibly coming off a bye.
    # to handle this later, probably will drop them, save those
    # as a variable, and then re-merge after getting rid of the other
    # null values.
    df = df.dropna()
    return df

def eval_model(df):
    df['score_ratio'] = round(df['actual_score'] / df['pred'],4)
    return df

def remove_outliers_btwn_ij(df, i=-1, j=5):
    s = df.loc[(df.score_ratio > i) & (df.score_ratio < j)]
    return s, i, j

def get_RMSE(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    return RMSE

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def invert_one_hot_encode(df, cols=None, sub_strs=None):
    df['Name'] = (df.iloc[:, 3:len(df)] == 1).idxmax(1).str.replace('Name_', "")
    subset = ['Week', 'DK salary', 'Oppt_pts_allowed_lw', 'Name']
    new_df = df[subset]
    return new_df

def get_current_year():
    from datetime import datetime
    today = datetime.today()
    datem = datetime(today.year, today.month, 1)
    return datem.year

def train_models():
    season = get_current_year()

    dataset = get_season_data(season)
    df = handle_nulls(dataset)
    def_df = df.loc[df.Pos == 'Def']
    def_df['fantasy_points_allowed_lw'] = 0
    df['Oppt_pts_allowed_lw'] = 0
    def_teams = [x for x in def_df['Team'].unique()]

    def_df['pred'] = 1
    def_df = def_df.rename(columns={'DK points': 'actual_score'})
    def_df

    for week in range(1,17):
        for team in def_teams:
            try:
                offense_df1 = df.loc[(df['Oppt']==team)&(df['Week']==week)]
                offense_df2 = df.loc[(df['Oppt']==team)&(df['Week']==week+1)]
                sum_ = offense_df1['DK points'].sum()
                def_df.loc[(df['Team']==team)&(df['Week']==week+1), 'fantasy_points_allowed_lw'] = sum_
                df.loc[(df['Oppt']==team)&(df['Week']==week+1), 'Oppt_pts_allowed_lw'] = sum_
            except:
                print('couldnt append data')
                pass
    df = df[df.Week != 1] # can't predict values for this week so just drop it
    X = df.drop(labels='DK points', axis=1)
    y = df['DK points']
    X2 = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state = 42)

    ab_reg = AdaBoostRegressor(**{'learning_rate': 0.02, 
                                  'loss': 'exponential', 
                                  'n_estimators': 100}) 
    gb_reg = GradientBoostingRegressor(**{'learning_rate': 0.05, 
                                          'max_depth': 3, 
                                          'max_features': 'auto', 
                                          'min_samples_leaf': 2})
    ab_reg.fit(X_train, y_train)
    gb_reg.fit(X_train, y_train)

    return ab_reg, gb_reg

def fix_names(name):
    name = name.split(' ', 1)
    name.reverse()
    name = ", ".join(name)
    return name

def get_dk_data():
    list_of_files = glob.glob("./csv's/dkdata/*.csv") 
    sorted_files = sorted(list_of_files, key=os.path.getctime)
    most_recent_dkdata = sorted_files[-1] 

    dk_df = pd.read_csv(most_recent_dkdata)
    drop_labels = []
    for col in dk_df:
        if 'Unnamed' in col:
            drop_labels.append(col)
    dk_df = dk_df.drop(drop_labels, axis=1)
    # print(dk_df)
    return dk_df

def fix_dk_data(dk_df):
    dk_df = dk_df.rename(columns={'TeamAbbrev': 'Team', 'Position':'Pos'})
    dk_df['Team'] = dk_df['Team'].apply(str.lower)
    dk_df['Oppt'] = dk_df['Oppt'].apply(str.lower)
    dk_df['Name'] = dk_df['Name'].apply(fix_names)
    return dk_df

year = get_current_year()
week = input("What week is it: ")
print(week)

dk_df = get_dk_data()
dk_data = fix_dk_data(dk_df)
dk_data_un = dk_data["Name"].unique()
# print(dk_data_un)
print(len(dk_data_un))

prev_data = get_ytd_season_data(year, int(week))
prev_data_un = prev_data["Name"].unique()
# print(prev_data_un)
print(len(prev_data_un))

count = 0
for i in range(len(prev_data_un)):
    for j in range(len(dk_data_un)):
        sim = similar(prev_data_un[i], dk_data_un[j])
        if sim > 0.87:
            print(sim)
            print(prev_data_un[i], "|", dk_data_un[j])
            count += 1
            time.sleep(0.2)
print(count)