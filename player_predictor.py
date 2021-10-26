import getopt, sys
import glob
import os
import random
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # to remove some warnings

from lineup_builder import Lineup
from dfs_data_scraper import scraper
from predictor_utils import *

year = get_current_year()
week = int(input("What week is it: "))
print(week)

# build and wrangle data
dk_df = get_dk_data()
dk_df_un = dk_df["Name"].unique()

def_df = build_defense_df(week)
def_df.to_csv(path_or_buf=f"./csv's/def_df's/most_recent_def_df.csv")

prev_data = get_ytd_season_data(year, int(week))
prev_data_un = prev_data["Name"].unique()

name_map = dict.fromkeys(dk_df_un)
for name in name_map.keys():
    possibilities = []
    choice = ''
    for i in range(len(prev_data_un)):    
        sim = similar(name, prev_data_un[i])
        if sim > 0.87:
            possibilities.append(prev_data_un[i])
    if len(possibilities) > 1:
        if name in possibilities:
            name_map[name] = name
            continue
        # edge cases get handled by human
        print(possibilities)
        idx = int(input(f"Which one looks right for {name}? "))
        choice = possibilities[idx-1]
        name_map[name] = choice
    else:
        try:
            name_map[name] = possibilities[0]
        except:
            pass
    
    if (name_map[name]) == None:
        name_map[name] = name

# Fix the names so they get the same
# labels from the ordinal encoder
for row in dk_df['Name']:
    if row in name_map.keys():
        dk_df.loc[dk_df['Name'] == row, 'Name'] = name_map[row]

# train the models
dk_df=dk_df.drop(columns=['ID', 'AvgPointsPerGame'])

ab_reg, gb_reg = train_models(week, prev_data)

# make predictions
prediction_df = predict_players(gb_reg, ab_reg, prev_data, dk_df, week)

# prediction_df = encode_features(encoders, prediction_df, method='decode')
    
pd.set_option("display.max_rows", None, "display.max_columns", 20)

# save predictions
prediction_df.to_csv(path_or_buf=f"./csv's/dkdata/predictions/dk_preds-week-{week}.csv")

recs = get_recommendations(week)
print("This week's good plays: \n")
print(recs)

# pick defenses
t1d, t2d = pick_def(week)
dk_df2 = get_dk_data().drop(columns=['ID'])
for i in range(len(t2d)):
    try:
        print("Tier 1 Def glance: \n")
        msg = dk_df2.loc[(dk_df2['Oppt'] == t1d[i])].head(10)
        if len(msg) == 0:
            print(f"Looks like {t1d[i].upper()} is on bye.")
        else:
            print(msg)
    except:
        pass
    try:
        print("Tier 2 Def glance: \n")
        msg = dk_df2.loc[(dk_df2['Oppt'] == t2d[i])].head(10)
        if len(msg) == 0:
            print(f"Looks like {t2d[i].upper()} is on bye.")
        else:
            print(msg)
    except:
        pass

txt_str = "This week's Good Plays: \n " \
    f'{recs} \n\n' \
    f'Tier 1 Defenses: \n' \
    f'{", ".join([x.upper() for x in t1d])} \n\n' \
    f'Tier 2 Defenses: \n' \
    f'{", ".join([x.upper() for x in t2d])} \n' \

print(txt_str)

text_file = open(f"./txt/{week}-{year}-pred.txt", "w")
n = text_file.write(txt_str)
text_file.close()