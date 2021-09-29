class Lineup:
    """ 
    takes the results of the model prediction (dataframe 
    with attached predictions) and builds out a few lineups 
    """
    def __init__(self, df, def_df, verbose=False):
        self.verbose = verbose
        self.df = df
        self.def_df = def_df[:15]
        self.current_salary = 100*1000
        self.no_duplicates = False
        self.top_lineups = []
        self.qbs = []
        self.rbs = []
        self.wrs = []
        self.tes = []
        self.flex = []
        self.defs = []
    
    def find_top_10(self, position):
        arr = []
        end_of_range = len(self.df.loc[self.df['Pos']==position])
        if position == 'Flex':
            position_df = self.df.loc[(self.df['Pos']=='RB')|(self.df['Pos']=='TE')|(self.df['Pos']=='WR')]
            end_of_range = (len(self.df.loc[self.df['Pos']=='RB'])+
                            len(self.df.loc[self.df['Pos']=='WR'])+
                            len(self.df.loc[self.df['Pos']=='TE']))
        elif position == 'Def':
            end_of_range = len(self.def_df)
            position_df = self.def_df
            position_df = position_df.sort_values(by='pred', ascending=False)
        else:
            position_df = self.df.loc[self.df['Pos']==position]
        
        # print(position_df)
        for row in range(0,end_of_range):
            player = {
                'name': position_df.iloc[row]['Name'],
                'h/a': position_df.iloc[row]['h/a'],
                'pos': position_df.iloc[row]['Pos'],
                'salary': position_df.iloc[row]['DK salary'],
                'pred_points': position_df.iloc[row]['pred'],
                'act_pts':position_df.iloc[row]['actual_score']
            }
            if len(arr) < end_of_range:
                arr.append(player)
            else: 
                break
        return arr
    
    def get_players(self):
        top_10_qbs = self.find_top_10(position='QB')
        top_10_rbs = self.find_top_10(position='RB')
        top_10_wrs = self.find_top_10(position='WR')
        top_10_tes = self.find_top_10(position='TE')
        top_10_flex = self.find_top_10(position='Flex')
        top_10_defs = self.find_top_10(position='Def')
        return top_10_qbs, top_10_rbs, top_10_wrs, top_10_tes, top_10_flex, top_10_defs
    
    def check_salary(self, lineup):
        current_salary = 0
        for keys in lineup.keys():
            current_salary += lineup[keys]['salary']
        return current_salary
    
    def reduce_salary(self, lineup):
        while self.current_salary > 50*1000:
            position_df = self.df
            greatest_salary = 0
            pos = 'none'
            pos_to_change = 'none'
            for key in lineup.keys():
                if lineup[key]['salary'] > greatest_salary:
                    greatest_salary = lineup[key]['salary']
                    pos = lineup[key]['pos'] # RB, TE, Def, etc.
                    pos_to_change = key # RB1 or WR2 or something like that
            if pos_to_change == 'Def':
                position_df = def_df
            elif pos_to_change == 'Flex':
                position_df = self.df.loc[(self.df['Pos']=='RB')|(self.df['Pos']=='TE')|(self.df['Pos']=='WR')]
            else:
                pass
    #             print(position_df)    
            new_player = (position_df.loc[(position_df.Pos == pos)&(position_df['DK salary'] < greatest_salary)]).sort_values(by='DK salary', ascending=False).head(1)
            player = {
                'name': new_player['Name'].values[0],
                'h/a': new_player['h/a'].values[0],
                'pos': new_player['Pos'].values[0],
                'salary': new_player['DK salary'].values[0],
                'pred_points': new_player['pred'].values[0],
                'act_pts':new_player['actual_score'].values[0]
            }
    #         print(player)    
            lineup[pos_to_change] = player
    #         print(lineup)
            self.current_salary = self.check_salary(lineup)
        return lineup
    
    def check_duplicates(self, lineup):
        rb1_name = lineup['RB1']['name']
        rb2_name = lineup['RB2']['name']
        flex_name = lineup['Flex']['name']
        wr1_name = lineup['WR1']['name']
        wr2_name = lineup['WR2']['name']
        wr3_name = lineup['WR3']['name']
        te_name = lineup['TE']['name']
        names = [flex_name, rb1_name, rb2_name, wr1_name, wr2_name, wr3_name, te_name]
        while len(names) > 1:
            if names[0] in names[1:-1]:
                return False
            else:
                names.pop(0)   
        return True
    
    def shuffle_players(self):
        lineup = {
            'QB': self.qbs[random.randrange(len(self.df.loc[self.df['Pos']=='QB']))],
            'RB1': self.rbs[random.randrange(len(self.df.loc[self.df['Pos']=='RB']))],
            'RB2': self.rbs[random.randrange(len(self.df.loc[self.df['Pos']=='RB']))],
            'WR1': self.wrs[random.randrange(len(self.df.loc[self.df['Pos']=='WR']))],
            'WR2': self.wrs[random.randrange(len(self.df.loc[self.df['Pos']=='WR']))],
            'WR3': self.wrs[random.randrange(len(self.df.loc[self.df['Pos']=='WR']))],
            'TE': self.tes[random.randrange(len(self.df.loc[self.df['Pos']=='TE']))],
            'Flex': self.flex[random.randrange(len(self.df.loc[self.df['Pos']=='RB'])+
                                               len(self.df.loc[self.df['Pos']=='WR'])+
                                               len(self.df.loc[self.df['Pos']=='TE']))],
            'Def': self.defs[random.randrange(len(self.def_df))]
        }
        return lineup
    
    def build_lineup(self,verbose=False):
        # in theory, because of the legwork done by the algorithm,
        # any lineup should be good as long as it abides by the
        # constraints of DraftKings' team structures. So for
        # now, this will just give us the lineups that fit within
        # the salary cap and team requirements
        
        self.verbose = verbose
        self.current_salary = 100*1000
        self.no_duplicates = False
        if len(self.qbs) < 1:
            self.qbs, self.rbs, self.wrs, self.tes, self.flex, self.defs = self.get_players()
        lineup = self.shuffle_players()
        
        while True:
            if self.verbose:
                print('======================')
                print(f"Salary: {self.current_salary}")
                print(f"No Duplicates: {self.no_duplicates}")
                print('======================')
            self.no_duplicates = self.check_duplicates(lineup)
            self.current_salary = self.check_salary(lineup)
            # fix duplicates first
            if self.no_duplicates == False:
                lineup = self.shuffle_players()
            # check salary, making sure it's between 45k and 50k
            if self.current_salary > 50*1000:
                try:
                    lineup = self.reduce_salary(lineup)
                except:
                    lineup = self.shuffle_players()
            self.no_duplicates = self.check_duplicates(lineup)
            self.current_salary = self.check_salary(lineup)
            
            if (self.current_salary <= 50*1000 
#             and self.current_salary >= 45*1000 
            and self.no_duplicates):
                # if everything looks good, end the 
                # loop and append the lineup
                break
                
        
        self.top_lineups.append(lineup)
        if len(self.top_lineups) % 5 == 0:
            print(f"Added lineup. Total lineups: {len(self.top_lineups)}")
    
