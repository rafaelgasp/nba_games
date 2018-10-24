import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import json
import requests
import pickle

def get_date_place(game_id, year = "2017"):
    #0021700784
    try:
        resp = requests.get(url="https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/" + str(year) + "/scores/gamedetail/"+ str(game_id) +"_gamedetail.json",
                            headers=headers)
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
                                          'COMMENT', 'MIN', 'MINUTES']):
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