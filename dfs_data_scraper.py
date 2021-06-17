# scraper 
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd
import lxml.html as lh
from lxml.html.clean import Cleaner
from pathlib import Path
from random import seed, random
import requests, time, os, sys


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
            # if the file already exists of the data, don't scrape for it
            location_string = f"./csv's/{i}/year-{i}-week-{j}-DK-player_data.csv"
            _file = Path(f'{location_string}')
            if _file.exists():
                pass
            # otherwise, make the url and scrape for it
            else:
                query_string = f"week={j}&year={i}&game=dk&scsv=1"
                urls.append(base_url + query_string)
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

def get_fantasy_data(url):
    from splinter import Browser
    from selenium import webdriver
    from settings import CHROMEDRIVER_DIR1, CHROMEDRIVER_DIR2
    # define the location of the Chrome Driver - CHANGE THIS!!!!!
    executable_path = {'executable_path': CHROMEDRIVER_DIR2}

    # Create a new instance of the browser, make sure we can see it (Headless = False)
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # define the components to build a URL
    method = 'GET'

    # build the URL and store it in a new variable
    p = requests.Request(method, url).prepare()
    myurl = p.url

    # go to the URL
    try:
        browser = Browser('chrome', **executable_path, headless=False, incognito=True, options=options)
        browser.visit(myurl)
    except:
        # if chrome auto updates and opening a browser fails, 
        # try a different webdriver version
        executable_path = {'executable_path': CHROMEDRIVER_DIR1}
        browser = Browser('chrome', **executable_path, headless=False, incognito=True, options=options)
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
    location_string = f"./csv's/{df['Year'][0]}/year-{df['Year'][0]}-week-{df['Week'][0]}-DK-player_data.csv"
    df.to_csv(path_or_buf=location_string)
    return 

def scraper():
    urls = build_urls()
    for link in urls:
        get_fantasy_data(link)

if __name__ == '__main__':
    setup_folders()
    # scraper()
    get_player_data(player_name="Ryan, Matt", week=5, year=2018)