import getopt, sys

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # to remove some warnings

from lineup_builder import Lineup
from dfs_data_scraper import scraper
from predictor_utils import *

year = get_current_year()
week = int(input("What week is it: "))
print(week)

# User can pass -s or --skip when calling script to 
# skip initial data scrape
# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first
argument_list = full_cmd_arguments[1:]

short_options = "s"
long_options = ["skip"]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-s", "--skip"):
        break
    else:
        # make sure all of the data is up to date
        scraper()

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
    for i in range(len(prev_data_un)):    
        sim = similar(name, prev_data_un[i])
        if sim > 0.87:
            possibilities.append(prev_data_un[i])
    if len(possibilities) > 1:
        print(possibilities)
        idx = int(input(f"Which one looks right for {name}? "))
        choice = possibilities[idx-1]
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

prediction_df = encode_features(encoders, prediction_df, method='decode')
    
pd.set_option("display.max_rows", None, "display.max_columns", 20)

# save predictions
prediction_df.to_csv(path_or_buf=f"./csv's/dkdata/predictions/dk_preds-week-{week}.csv")

recs = get_recommendations(week)
print("This week's good plays: \n")
print(recs)

# pick defenses
t1d, t2d = pick_def()
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