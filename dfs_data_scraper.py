# scraper 
from bs4 import BeautifulSoup
import csv
from io import StringIO
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from random import seed, random
import requests, time, os, sys
from shutil import move

from settings import DL_DIR, DK_NAME, DK_PW
from webdriver_updater import get_updates

from setup_folders import setup_folders

def get_current_year():
    from datetime import datetime
    today = datetime.today()
    datem = datetime(today.year, today.month, 1)
    return datem.year

def build_urls():
    # url model
    # http://rotoguru1.com/cgi-bin/fyday.pl?week=17&year=2019&game=dk&scsv=1

    # setup weeks and years
    weeks = [i for i in range(1,18)]
    years = [i for i in range(2014, get_current_year()+1)]

    # create scraping urls
    base_url = "http://rotoguru1.com/cgi-bin/fyday.pl?"
    urls = []
    for i in years:
        for j in weeks:
            try:
                # if the file already exists of the data, don't scrape for it
                location_string = f"./csv's/{i}/year-{i}-week-{j}-DK-player_data.csv"
                _file = Path(f'{location_string}')
                if _file.exists():
                    pass
                # otherwise, make the url and scrape for it
                else:
                    query_string = f"week={j}&year={i}&game=dk&scsv=1"
                    urls.append(base_url + query_string)
            except IndexError:
                # if this runs mid-season, we don't want to keep
                # scanning for other weeks
                break
            except:
                pass
    return urls

def get_player_data(player_name, week=0, year=get_current_year()):
    """player_name should be last name first. If week is left out, then get season data. Default year is current year. """
    # url model
    # https://www.pro-football-reference.com/players/R/RodgAa00/gamelog/2020/
    # looks like we need a last initial, first 4 letters of last name, and first 2 letters of first name
    last_initial = player_name[0]
    last_name_first_letters = player_name[0:4]
    first_name_first_letters = player_name.strip().split(',')[1][1:3]
    
    # build url
    url = f"https://www.pro-football-reference.com/players/{last_initial}/{last_name_first_letters}{first_name_first_letters}00/gamelog/{year}/"
    print(url)
    # hoping I don't need browser automation for this...
    df = pd.read_html(url, index_col=0)
    new_index = [i for i in range(0,len(df[0])+1)]
    df = df[0].reindex(new_index)
    df = df.drop([0], axis=0)
    # print(df.head())
    if week > 0:
        df_sub = df.iloc[week-1]
        print(df_sub)
        return df_sub
    print(df)
    return df

def get_browser(tries=0):
    from splinter import Browser
    from selenium import webdriver
    from settings import CHROMEDRIVER_DIR1, CHROMEDRIVER_DIR2, CHROMEDRIVER_DIR3
    # Create a new instance of the browser, make sure we can see it (Headless = False)
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    executable_path = {}
    if tries == 0:
        executable_path = {'executable_path': CHROMEDRIVER_DIR3}
    elif tries == 1:
        executable_path = {'executable_path': CHROMEDRIVER_DIR2}
    elif tries == 2:
        executable_path = {'executable_path': CHROMEDRIVER_DIR1}
    else:
        print("Couldn't build browser.")
        return
    try:
        browser = Browser('chrome', **executable_path, headless=False, incognito=True, options=options)
        return browser
    except:
        tries += 1
        get_browser(tries)
        

def get_dk_data():
    # https://www.draftkings.com/lineup/upload#

    # dl button html:
    # <a href="/bulklineup/getdraftablecsv?draftGroupId=56618" class="dk-btn dk-btn-success dk-btn-icon pull-right" data-download-template="1">
    # <span class="icon-download"></span> DOWNLOAD</a>

    # define the components to build a URL
    method = 'GET'

    # build the URL and store it in a new variable
    p = requests.Request(method, 'https://www.draftkings.com/lineup/upload#').prepare()
    myurl = p.url

    browser= get_browser()
    
    # go to the URL
    browser.visit(myurl)
    seed(1)
    time.sleep(random()+1)

    # log in
    login_btn = browser.find_by_text('Log In')
    login_btn.click()
    time.sleep(3)
    browser.find_by_name('username').click()
    time.sleep(1)
    browser.find_by_name('username').fill(DK_NAME)
    time.sleep(1)
    browser.find_by_name('password').click()
    time.sleep(1)
    browser.find_by_name('password').fill(DK_PW)
    
    time.sleep(1)
    login_btn = browser.find_by_text('Log In')[1]
    login_btn.click()
    
    while True:
        try:
            time.sleep(25)
            dl_button = browser.links.find_by_partial_href('bulklineup/getdraftablecsv')
            time.sleep(1)
            dl_button.click()
            time.sleep(5)
            break
        except:
            pass

    browser.quit()

    download_path = DL_DIR
    new_path = os.getcwd()
    old_file_name = download_path+'/DKSalaries.csv'
    list_of_files = [glob.glob("./csv's/dkdata/*")]
    seq = len(list_of_files)
    new_file_name = new_path+f'/csv\'s/dkdata/DKSalaries-{seq}.csv'
    move(old_file_name, new_file_name)
    fix_dk_salaries(new_file_name)
    return

def diff_sets(game_info, team):
    arr_game_info = game_info.split('@')
    arr_team = [team]
    set_game_info = set(arr_game_info)
    set_team = set(arr_team)
    diff = set_game_info.difference(set_team)
    return ''.join(diff)

def fix_dk_salaries(file_name):
    arr = []
    with open(file_name, newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            # print(len(line))
            if len(line) > 12:
                arr.append(line)
    cols = arr.pop(0)
    arr = np.array(arr)
    csv_file.close()
    
    df = pd.DataFrame(data=arr, columns=cols)
    df['Game Info'] = df['Game Info'].apply(lambda val: val.split(' ')[0])
    df['h/a'] = df.apply(lambda row: 'a' if row['Game Info'].startswith(row['TeamAbbrev']) else 'h', axis=1)
    df['Oppt'] = df.apply(lambda row: diff_sets(row['Game Info'], row['TeamAbbrev']), axis=1)
    df = df.drop(labels=['Name + ID', 'Roster Position', 'Game Info'], axis=1)
    print(df.head(15))
    file_name_rewrite = file_name.split('/')
    file_name_rewrite[-1] = 'fixed_'+file_name_rewrite[-1]
    new_file_name = '/'.join(file_name_rewrite)
    df.to_csv(path_or_buf=new_file_name)
    os.remove(file_name)

def get_fantasy_data(url):

    # define the components to build a URL
    method = 'GET'

    # build the URL and store it in a new variable
    p = requests.Request(method, url).prepare()
    myurl = p.url
           
    # go to the URL
    browser = get_browser()
    browser.visit(myurl)
    
    seed(1)
    time.sleep(random()+1)
    
    # add a little randomness to using the page
    time.sleep(random()+random()*10)

    html = browser.html

    browser.quit()

    soup = BeautifulSoup(html, 'lxml')
    csv = StringIO(soup.pre.get_text())
    df = pd.read_csv(csv, sep=';')
    df = df.drop(columns='GID')
    # print(df.head())
    try:
        location_string = f"./csv's/{df['Year'][0]}/year-{df['Year'][0]}-week-{df['Week'][0]}-DK-player_data.csv"
        df.to_csv(path_or_buf=location_string)
    except:
        raise Exception("Can't make file with fantasy data.")
    return 

def scraper():
    setup_folders()
    get_dk_data()
    urls = build_urls()
    for link in urls:
        try:
        # sometimes the url doesn't exist yet...
            get_fantasy_data(link)
        except:
        # so we just skip the rest of the season
            break
    get_updates()

if __name__ == '__main__':
    scraper()
    # get_player_data(player_name="Ryan, Matt", week=5, year=2018)