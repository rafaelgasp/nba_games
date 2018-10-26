import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import json
import requests
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
    except Exception as e:
        print(e)
        data = np.nan
        place = np.nan
    
    return(data, place)

def get_df_nba_json(resp_json, date_place = False, rs=0):
    dict_resp = resp_json['resultSets'][rs]
    df_resp = pd.DataFrame(dict_resp["rowSet"])
    df_resp.columns = dict_resp["headers"]
    
    if(date_place):
        game_date, game_place = get_date_place(df_resp.GAME_ID.iloc[0])

        df_resp["GAME_DATE"] = np.repeat(game_date, len(df_resp))
        df_resp["GAME_PLACE"] = np.repeat(game_place, len(df_resp))

        teams = df_resp.TEAM_ABBREVIATION.unique()

        df_resp["GAME"] = np.repeat(teams[0] + " @ " + teams[1] + " " + game_date, len(df_resp))
    
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
    for i in range(start_at, 1231):
        gameid = "002" + year[-2:] + "0" + ('{0:0>4}'.format(i))
        
        game_date = get_date_place(gameid, year = "2018")[0]
        print((gameid, game_date), end="\r")
        if(datetime.strptime(game_date, '%Y-%m-%d') > max_date):
            break
        
        game_ids.append(gameid)
    return(game_ids)

def get_nba_stats_data(game_ids, lista_sites = ["traditional", "advanced", "scoring", 
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

    for game_id in game_ids:

        df_tipos = []
        df_tipos_jogo = []

        try:
            game_date, game_place = get_date_place(game_id, year)
    
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

            teams = df_resp.TEAM_ABBREVIATION.unique()
            game_str = teams[0] + " @ " + teams[1] + " " + game_date

            df_resp["GAME"] = np.repeat(game_str, len(df_resp))
            df_resp_jogo["GAME"] = np.repeat(game_str, len(df_resp_jogo))

            df_full.append(df_resp.set_index("GAME"))
            df_full_jogo.append(df_resp_jogo.set_index("GAME"))

            #pickle.dump(df_full, open("df_full.p", "wb"))
            #pickle.dump(df_full_jogo, open("df_full_jogo.p", "wb"))
        except Exception as e:
            erros.append(game_id)
            print(e)
            time.sleep(3)
    
    return(df_full, df_full_jogo, erros)