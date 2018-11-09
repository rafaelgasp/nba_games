import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import json
import requests
from requests import RequestException
import pickle

NBA_STATS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
}

def get_date_place(game_id, year = "2017"):
    #0021700784
    try:
        resp = requests.get(url="https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/" + str(year) + "/scores/gamedetail/"+ str(game_id) +"_gamedetail.json",
                            headers=NBA_STATS_HEADERS)
        data = resp.json()["g"]["gdte"]
        place = resp.json()["g"]["an"]
        team_away = resp.json()["g"]["gcode"][9:-3]
        team_home = resp.json()["g"]["gcode"][-3:]
    except Exception as e:
        print(e)
        data = np.nan
        place = np.nan
        team_away = np.nan
        team_home = np.nan
    
    return(data, place, team_away, team_home)

def get_df_nba_json(resp_json, date_place = False, rs=0):
    dict_resp = resp_json['resultSets'][rs]
    df_resp = pd.DataFrame(dict_resp["rowSet"])
    if(df_resp.empty):
        df_resp = pd.DataFrame(pd.np.empty((0,len(dict_resp["headers"]))))
    df_resp.columns = dict_resp["headers"]
    
    if(date_place):
        game_date, game_place, team_away, team_home = get_date_place(df_resp.GAME_ID.iloc[0])

        df_resp["GAME_DATE"] = np.repeat(game_date, len(df_resp))
        df_resp["GAME_PLACE"] = np.repeat(game_place, len(df_resp))

        df_resp["GAME"] = np.repeat(team_away + " @ " + team_home + " " + game_date, len(df_resp))
    
    if "TEAM_NAME" in df_resp.columns:
        df_resp = df_resp.sort_values("TEAM_NAME")
    else:
        if "TEAM_NICKNAME" in df_resp.columns:
            df_resp = df_resp.sort_values("TEAM_NICKNAME")
        else:
            df_resp = df_resp.sort_values("TEAM_ABBREVIATION")
        
    return(df_resp)

def junta_df_tipos(df_tipos, cols_drop = ['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY', 
                                          'PLAYER_ID', 'PLAYER_NAME', 'START_POSITION', 
                                          'COMMENT', 'MIN', 'MINUTES'],
                              lista_sites = ["traditional", "advanced", "scoring", 
                                                "misc", "usage", "fourfactors", "playertrack", 
                                                "hustle", "defensive"]):
    resp = df_tipos[0]
    for i in range(1, len(df_tipos)):
        junta = df_tipos[i]
        colunas_repetidas = list(set(junta.columns).intersection(resp.columns))
        
        junta = junta.drop(cols_drop, axis=1, errors="ignore")
        junta.columns = [str(col) + '_' + lista_sites[i]
                         if col in colunas_repetidas else str(col) 
                         for col in junta.columns]
        
        resp = resp.merge(junta, how="left", 
                          left_index=True, right_index=True)
    return(resp)

def get_list_gameids(max_date = datetime.today() - timedelta(1), year = '2018', start_at=1):
    game_ids = []
    for i in range(start_at, 1230 + 1):
        gameid = "002" + year[-2:] + "0" + ('{0:0>4}'.format(i))
        
        game_date = get_date_place(gameid, year = "2018")[0]
        print((gameid, game_date), end="\r")
        if(datetime.strptime(game_date, '%Y-%m-%d') >= max_date):
            break
        
        game_ids.append(gameid)
    return(game_ids)

def get_game_infos_dates(games_ids):
    year = "20" + games_ids[0][3:5]
    
    resp = {
        "GAME": [],
        "DATE": [],
        "PLACE": [],
        "team_home": [],
        "team_away": [],
    }
    
    for game_id in games_ids:
        game_date, game_place, team_away, team_home = get_date_place(game_id, year)
        print(team_away + " @ " + team_home + " " + game_date + "              ", end="\r")
        resp["GAME"].append(team_away + " @ " + team_home + " " + game_date)
        resp["DATE"].append(game_date)
        resp["PLACE"].append(game_place)
        resp["team_away"].append(team_away)
        resp["team_home"].append(team_home)
        
    resp = pd.DataFrame(resp)
    resp.DATE = pd.to_datetime(resp.DATE)
    return(resp)
        

def get_nba_stats_data(games_ids, lista_sites = ["traditional", "advanced", "scoring", 
                                                "misc", "usage", "fourfactors", "playertrack", 
                                                "hustle", "defensive"]):    
    params = {
        'EndPeriod':10,
        'EndRange':28800,
        'GameID':'',
        'RangeType':0,
        'Season':2018-19,
        'SeasonType':'Regular+Season',
        'StartPeriod':1,
        'StartRange':0
    }
    
    df_full = []
    df_full_jogo = []
    erros = []
    
    year = "20" + games_ids[0][3:5]

    for game_id in games_ids:

        df_tipos = []
        df_tipos_jogo = []

        try:
            game_date, game_place, team_away, team_home = get_date_place(game_id, year)
    
            for site in lista_sites:
                if(site == "hustle"):
                    url = "https://stats.nba.com/stats/hustlestatsboxscore"
                    rs = 1

                elif(site == "defensive"):
                    url = "https://stats.nba.com/stats/boxscore" + site
                    rs = -1

                else:
                    url = "https://stats.nba.com/stats/boxscore" + site + "v2"
                    rs = 0

                print(game_id + " " + str(game_date) + " - " + site + "              ", end="\r")
                #print(game_id + " " + str(game_date) + " - " + site)
                
                params["GameID"] = game_id

                resp = requests.get(url=url, params=params, headers=NBA_STATS_HEADERS)
                #time.sleep(0.5)

                if(rs == 0):
                    df_tipos.append(get_df_nba_json(resp.json(), rs=0))
                    df_tipos_jogo.append(get_df_nba_json(resp.json(), rs=1))
                elif(rs == 1):
                    df_tipos.append(get_df_nba_json(resp.json(), rs=1))
                    df_tipos_jogo.append(get_df_nba_json(resp.json(), rs=2))
                elif(rs == -1):
                    df_tipos.append(get_df_nba_json(resp.json(), rs=0))

            df_resp = junta_df_tipos(df_tipos, lista_sites=lista_sites)
            df_resp_jogo = junta_df_tipos(df_tipos_jogo, lista_sites=lista_sites)

            df_resp["GAME_DATE"] = np.repeat(game_date, len(df_resp))
            df_resp["GAME_PLACE"] = np.repeat(game_place, len(df_resp))

            df_resp_jogo["GAME_DATE"] = np.repeat(game_date, len(df_resp_jogo))
            df_resp_jogo["GAME_PLACE"] = np.repeat(game_place, len(df_resp_jogo))

            game_str = team_away + " @ " + team_home + " " + game_date

            df_resp["GAME"] = np.repeat(game_str, len(df_resp))
            df_resp_jogo["GAME"] = np.repeat(game_str, len(df_resp_jogo))

            df_full.append(df_resp.set_index("GAME"))
            df_full_jogo.append(df_resp_jogo.set_index("GAME"))

            #pickle.dump(df_full, open("df_full.p", "wb"))
            #pickle.dump(df_full_jogo, open("df_full_jogo.p", "wb"))
        except RequestException as e:
            print(e)
        #except Exception as e:
        #    erros.append(game_id)
        #    print(e)
            #time.sleep(3)
    
    return(df_full, df_full_jogo, erros)

# ODDS PORTAL
def trata_df_odds(table_html):
    df_odds = pd.read_html(table_html)
    df_odds = pd.concat(df_odds)
    df_odds[0] = df_odds[0].shift(1)
    
    df_odds.columns = ["date", "game", "result", "odds_home", "odds_away", "B's"]
    df_odds.dropna(subset=(["odds_home", "B's"]), inplace=True)
    
    df_odds["date"] = [x if len(x) > 5 else np.nan for x in df_odds["date"]]
    df_odds["date"].fillna(method='ffill', inplace=True)
    df_odds["data"] = [datetime.strptime(x.split(" - ")[0], '%d %b %Y') for x in df_odds["date"]]
    
    return(df_odds)

def parse_home_team(game_str):
    for key in de_para_siglas.keys():
        game_str = game_str.replace(key, de_para_siglas[key])
    
    game_str = game_str.replace("-", "@")
    spl =  game_str.split(" @ ")
    return(game_str, spl[0], spl[1])

def get_odds():
    "https://www.oddsportal.com/basketball/usa/nba/"
