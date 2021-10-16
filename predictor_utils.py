import glob
import os
import random
from difflib import SequenceMatcher
import sys

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # to remove some warnings

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# features for encoding
features = ['Salary', 'Name', 'Team', 'Oppt', 'Pos', 'h/a']
# ordinal encoders and scalers for models
name_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99999)
team_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99999)
h_a_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99999)
oppt_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99999)
pos_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99999)
encoders = {
    'Name': name_enc,
    'Team':team_enc,
    'h/a':h_a_enc,
    'Oppt':oppt_enc,
    'Pos':pos_enc
}

name_sc = StandardScaler()
team_sc = StandardScaler()
h_a_sc = StandardScaler()
oppt_sc = StandardScaler()
pos_sc = StandardScaler()
salary_sc = StandardScaler()
scalers = {
    'Name': name_sc,
    'Team':team_sc,
    'h/a':h_a_sc,
    'Oppt':oppt_sc,
    'Pos':pos_sc,
    'Salary': salary_sc
}

# Mapper for team names, only has the ones that are different
team_map = {
    'no' : 'nor',
    'kc' : 'kan',
    'gb' : 'gnb',
    'sf' : 'sfo',
    'tb' : 'tam',
    'jax': 'jac',
    'lv' : 'lvr',
    'oak' : 'lvr',
    'ne' : 'nwe'
}

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
    df['Team'] = df['Team'].apply(str.lower).apply(lambda x: team_map[x] if x in team_map.keys() else x)
    df['Oppt'] = df['Oppt'].apply(str.lower).apply(lambda x: team_map[x] if x in team_map.keys() else x)
    df = df.rename(columns={'DK salary': 'Salary'})
    return df

def get_season_data(year, drop_year=True):
    """ get entire season of data """
    df = get_weekly_data(1,year)
    for week in range(2,17):
        try:
            df = df.append(get_weekly_data(week, year), ignore_index=True)
        except:
            print("No data for week: "+str(week))
            break
    if drop_year:
        df = df.drop(['Unnamed: 0', 'Year'], axis=1)
    else:
        df = df.drop(['Unnamed: 0'], axis=1)
    df = df.rename(columns={'DK salary': 'Salary'})
    return df

def encode_features(encs, df, method=None):
    """ encodes categorical data. default method is fit_transform. 
    other options are fit, transform, and decode (inverse_transform) """
    for feature in features:
        if feature == 'Salary':
            continue
        single_enc = encs[feature]
        arr = np.array(df[feature])
        arr = arr.reshape((-1,1))
        res= ''
        if method == 'fit':
            res = single_enc.fit(arr)
        elif method == 'transform':
            res = single_enc.transform(arr)
        elif method == 'decode':
            res = single_enc.inverse_transform(arr)
        else: 
            res = single_enc.fit_transform(arr)

        encs[feature] = single_enc
        df[feature] = res
    return df

def scale_features(sc, df, method=None):
    """ scales continuous data. default method is fit_transform. 
    other options are fit, transform, and decode (inverse_transform) """
    for feature in features:
        single_sc = sc[feature]
        arr = np.array(df[feature])
        arr = arr.reshape((-1,1))
        res = ''
        if method == 'fit':
            res = single_sc.fit(arr)
        elif method == 'transform':
            res = single_sc.transform(arr)
        elif method == 'decode':
            res = single_sc.inverse_transform(arr)
        else: 
            res = single_sc.fit_transform(arr)

        sc[feature] = single_sc
        df[feature] = res
    return df

def handle_nulls(df):
    # players that have nulls for any of the columns are 
    # extremely likely to be under performing or going into a bye.
    # the one caveat is that some are possibly coming off a bye.
    # to handle this later, probably will drop them, save those
    # as a variable, and then re-merge after getting rid of the other
    # null values.
    df = df.dropna()
    return df

def similar(a, b):
    """ used to see level of similarity between 2 strings. """
    return SequenceMatcher(None, a, b).ratio()

def get_current_year():
    from datetime import datetime
    today = datetime.today()
    datem = datetime(today.year, today.month, 1)
    return datem.year

def get_extra_cols(prev_df, dk_df, week):
    """ Adds extra columns to each df for the purpose of teaching the model
    which defenses are really good and which aren't all that great. Calculates
    average fantasy points allowed by defenses to players from the previous 
    weeks """
    def_df = dk_df.loc[(dk_df.Pos == 'DST')|(dk_df.Pos == 'Def')]
    def_df['fantasy_points_allowed_lw'] = 0
    dk_df['Oppt_pts_allowed_lw'] = 0
    dk_df['Week'] = week
    def_teams = [x for x in def_df['Team'].unique()]

    for team in def_teams:
        try:
            offense_df = prev_df.loc[(prev_df['Oppt']==team)&(prev_df['Week']==week-1)]
            # sometimes a bye week messes with the above,
            # so go back another week
            if len(offense_df) <= 1:
                offense_df = prev_df.loc[(prev_df['Oppt']==team)&(prev_df['Week']==week-2)]
            sum_ = offense_df['DK points'].sum()
            def_df.loc[(prev_df['Team']==team)&(prev_df['Week']==week-1), 'fantasy_points_allowed_lw'] = sum_
            dk_df.loc[(dk_df['Oppt']==team)&(dk_df['Week']==week), 'Oppt_pts_allowed_lw'] = sum_
        except:
            print('couldnt append data: ', sys.exc_info())
            pass
    return dk_df

# needs refactoring
def train_models(week, prev_df):
    df = prev_df
    def_df = df.loc[df.Pos == 'Def']
    def_df['fantasy_points_allowed_lw'] = 0
    df['Oppt_pts_allowed_lw'] = 0
    def_teams = [x for x in def_df['Team'].unique()]

    def_df['pred'] = 1
    def_df = def_df.rename(columns={'DK points': 'actual_score'})
    def_df

    for _week in range(1,week):
        for team in def_teams:
            # print(team)
            try:
                # this seems to be the section causing 
                # problems with the raiders
                offense_df1 = df.loc[(df['Oppt']==team)&(df['Week']==_week)]
                if len(offense_df1) <= 1 and week > 1:
                    offense_df1 = df.loc[(df['Oppt']==team)&(df['Week']==_week-1)]
                # print("len of offense_df1: ", len(offense_df1), "team: ", team)
                sum_ = offense_df1['DK points'].sum()
                def_df.loc[(df['Team']==team)&(df['Week']==_week+1), 'fantasy_points_allowed_lw'] = sum_
                df.loc[(df['Oppt']==team)&(df['Week']==_week+1), 'Oppt_pts_allowed_lw'] = sum_
            except:
                print('couldnt append data. train models.')
                print('error: ', sys.exc_info())
                pass
    df = df[df.Week != 1] # no 'last week', so can't calculate values
    X = df.drop(labels=['DK points', 'Week'], axis=1)
    y = df['DK points']
    X2 = X.copy()
    X2 = encode_features(encoders, X2)
    X2 = scale_features(scalers, X2)

    ab_reg = AdaBoostRegressor(**{'learning_rate': 0.02, 
                                  'loss': 'exponential', 
                                  'n_estimators': 100}) 
    gb_reg = GradientBoostingRegressor(**{'learning_rate': 0.05, 
                                          'max_depth': 3, 
                                          'max_features': 'auto', 
                                          'min_samples_leaf': 2})
    ab_reg.fit(X2, y)
    gb_reg.fit(X2, y)

    return ab_reg, gb_reg

def fix_names(name):
    """ reformats names between to be the same in dataframes"""
    name = name.split(' ', 1)
    name.reverse()
    name = ", ".join(name)
    name = name.lstrip(", ")
    return name

def get_dk_data():
    """ grabs the most recent download of data from Draft Kings"""
    list_of_files = glob.glob("./csv's/dkdata/*.csv") 
    sorted_files = sorted(list_of_files, key=os.path.getctime)
    most_recent_dkdata = sorted_files[-1] 

    dk_df = pd.read_csv(most_recent_dkdata)
    drop_labels = []
    for col in dk_df:
        if 'Unnamed' in col:
            drop_labels.append(col)
    dk_df = dk_df.drop(drop_labels, axis=1)
    dk_df = dk_df.rename(columns={'TeamAbbrev': 'Team', 'Position':'Pos'})
    # next two lines working as intended up to here
    dk_df['Team'] = dk_df['Team'].apply(str.lower).apply(lambda x: team_map[x] if x in team_map.keys() else x)
    dk_df['Oppt'] = dk_df['Oppt'].apply(str.lower).apply(lambda x: team_map[x] if x in team_map.keys() else x)
    dk_df['Name'] = dk_df['Name'].apply(fix_names)
    dk_df['Pos'] = dk_df['Pos'].apply(lambda x: 'Def' if x == 'DST' else x)
    return dk_df

def build_defense_df(week, dk_df=None):
    """ further breaks down defenses by pts fantasy points
    allowed to each position """
    year = get_current_year()
    df = get_ytd_season_data(year, week)
    def_df = df.loc[df.Pos == 'Def']
    team_names = [x for x in def_df.Oppt.unique()]
    new_cols = ['avg_pts_to_qb', 'avg_pts_to_rb', 'avg_pts_to_wr', 'avg_pts_to_te']
    for col in new_cols:
        def_df[col] = 0
    weeks = df.Week.unique()
    pos = ['QB', 'RB', 'WR', 'TE']
    for team in team_names:
        df_temp = pd.DataFrame(columns=df.columns)
        for i in range(4):
            for week_ in weeks:
                df_week = df.loc[(df.Week == week_)&(df.Pos == pos[i])]
                df_week_team = df_week.loc[df_week.Oppt == team]
                df_temp = df_temp.append(df_week_team)
            df_temp_pos = df_temp.loc[df_temp.Pos == pos[i]]
            # this average is YTD average points a def is giving
            # to any one position
            for week_ in weeks:
                avg_ = df_temp_pos.loc[df_temp_pos.Week <= week_, 'DK points'].sum() / week_
                def_df.loc[(def_df.Team == team)&(def_df.Week == week_), new_cols[i]] = avg_
    return def_df

def predict_players(model1, model2, prev_df, dk_df, week):
    df = get_extra_cols(prev_df, dk_df, week)
    df = encode_features(encoders, df, method="transform")
    df = scale_features(scalers, df, method="transform")
    df = df.drop(columns='Week')
    y_pred = model1.predict(df)
    df['pred'] = y_pred
    df = scale_features(scalers, df, method="decode")
    df_filtered = df.loc[df['pred']>=10]

    df_filtered = df_filtered.drop(columns='pred')
    df_filtered = scale_features(scalers, df_filtered, method="transform")

    y_pred2 = model2.predict(df_filtered)
    df_filtered['pred'] = y_pred2
    df_filtered = scale_features(scalers, df_filtered, method="decode")
    return df_filtered

def get_recommendations(week, high_thresh=15, low_thresh=10):
    """ takes the results of the prediction and prints 'good'
    players to the terminal """
    file_path = f"./csv's/dkdata/predictions/dk_preds-week-{week}.csv"
    df = pd.read_csv(file_path)
    pd.set_option("display.max_rows", None, "display.max_columns", 20)
    df = df.loc[df.Name.isnull() == False]

    file_path = "./csv's/def_df's/most_recent_def_df.csv"
    def_df = pd.read_csv(file_path)

    # figure out which teams are giving up the most to qb's
    qb_df = (def_df.loc[(def_df.avg_pts_to_qb > high_thresh)]
                .drop(columns=['avg_pts_to_rb', 'avg_pts_to_wr', 'avg_pts_to_te'])
                .sort_values(by='avg_pts_to_qb', ascending=False).head(15))
    # sort by name to determine frequency of teams,
    # higher frequency = weaker to that position
    qb_df.sort_values(by='Name')
    qb_counts = qb_df.Team.value_counts()

    # figure out which teams are giving up the most to rb's
    rb_df = (def_df.loc[(def_df.avg_pts_to_rb > high_thresh)]
                .drop(columns=['avg_pts_to_qb', 'avg_pts_to_wr', 'avg_pts_to_te'])
                .sort_values(by='avg_pts_to_rb', ascending=False).head(15))
    # sort by name to determine frequency of teams,
    # higher frequency = weaker to that position
    rb_df.sort_values(by='Name')
    rb_counts = rb_df.Team.value_counts()

    # figure out which teams are giving up the most to wr's
    wr_df = (def_df.loc[(def_df.avg_pts_to_wr > high_thresh)]
                .drop(columns=['avg_pts_to_qb', 'avg_pts_to_rb', 'avg_pts_to_te'])
                .sort_values(by='avg_pts_to_wr', ascending=False).head(15))
    # sort by name to determine frequency of teams,
    # higher frequency = weaker to that position
    wr_df.sort_values(by='Name')
    wr_counts = wr_df.Team.value_counts()

    # figure out which teams are giving up the most to te's
    te_df = (def_df.loc[(def_df.avg_pts_to_te > low_thresh)]
                .drop(columns=['avg_pts_to_qb', 'avg_pts_to_rb', 'avg_pts_to_wr'])
                .sort_values(by='avg_pts_to_te', ascending=False).head(15))
    # sort by name to determine frequency of teams,
    # higher frequency = weaker to that position
    te_df.sort_values(by='Name')           
    te_counts = te_df.Team.value_counts() 

    total_counts = [qb_counts, rb_counts, wr_counts, te_counts]
    pos = ['qb', 'rb', 'wr', 'te']
    def read_counts(array):
        counts = {}
        for i in range(4):
            counts[pos[i]] = array[i][:3]
        return counts

    # this is the total times a def has given 18+
    # points (12+ in the case of TE's) up to any
    # defense. The higher the numer, the more frequent
    # that happens.
    count_dict = read_counts(total_counts)
    recs = pd.DataFrame(columns=['Name', 'Pos', 'Salary', 'Team', 'h/a', 'Oppt', 'pred'])
    for key in count_dict.keys():
        # print("Pick these", key + "'s:")
        for i in range(3):
            bad_def = count_dict[key].index[i]
            good_play = df.loc[(df.Oppt == bad_def)&(df.Pos == key.upper())].drop(columns=['Unnamed: 0', 'Oppt_pts_allowed_lw'])
            if len(good_play) > 0:
                # print(good_play)
                recs = recs.append(good_play)
        # print('=====')
    return recs

def pick_def():
    """ algo has trouble picking defenses, so this is based of just analysis. defenses 
    that most frequently appear by allowing the fewest fantasy points per game 
    will be the top defenses to pick """
    file_path = "./csv's/def_df's/most_recent_def_df.csv"
    def_df = pd.read_csv(file_path).drop(columns=['Unnamed: 0', 'Name', 'h/a']) 
    def_df = (def_df
                .sort_values(by='DK points', ascending=False)
                .head(int(len(def_df) / 10)))
    def_df_counts = def_df['Team'].value_counts()
    def_df_counts_idx = def_df_counts.index
    # print(def_df)

    tier_1_defs = []
    tier_2_defs = []
    match1 = -10
    match2 = -10
    for x, y in zip(def_df_counts, def_df_counts_idx):
        if x > match1:
            match1 = x
            tier_1_defs.clear()
            tier_1_defs.append(y)
        elif x == match1:
            tier_1_defs.append(y)

        if x < match1 and x > match2:
            match2 = x
            tier_2_defs.append(y)
        elif x == match2:
            tier_2_defs.append(y)

    return tier_1_defs, tier_2_defs