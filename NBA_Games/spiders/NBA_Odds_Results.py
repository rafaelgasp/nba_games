import scrapy

import re
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from time import gmtime, strftime
import time
import datetime
import pandas as pd

class NBAOddsResults(scrapy.Spider):
	name = 'nba_odds_results'

	handle_httpstatus_list = [500]

	anos = ["2017-2018", "2016-2017", "2015-2016" ]

	de_para_siglas = pd.read_csv('de_para_siglas.csv').set_index('nome').to_dict()['sigla']

	current_dir = "C:/Users/rafae/Google Drive/Projetos/NBA_Games/"

	base_url = ["http://www.oddsportal.com/basketball/usa/nba-2017-2018/results/page/1/"]
	#base_url = "http://www.oddsportal.com/basketball/usa/nba-"

	df = pd.DataFrame(columns=["game", "home_team", "home_team_odd", "away_team", ])

	def start_requests(self):	
		for url in self.base_url:
			yield scrapy.Request(url, callback=self.parse, dont_filter=True,
				headers={
                         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                         "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Mobile Safari/537.36",
                         "Accept-Encoding": "gzip, deflate",
						 "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
						 "Cache-Control": "max-age=0",
						 "Connection": "keep-alive",
                         "Host": "www.oddsportal.com",
						 "Upgrade-Insecure-Requests": 1,
                         "Origin": "www.oddsportal.com",
                         "Cookie":"op_state=1; op_testCountryCode=0; op_oddsportal=pgl4ls3mldd6sp9dtqgk9ha3u0; op_cookie-test=ok; op_last_id=48; op_user_time_zone=0; op_user_full_time_zone=31; op_user_cookie=884846553; op_user_hash=de0e1a9bdcf86da558475e3f227b2938; op_user_time=1534082693; _ga=GA1.2.1536448644.1534082694; _gid=GA1.2.275143357.1534082694; _gat_UA-821699-19=1; _gat_UA-821699-50=1",
						 "Access-Control-Allow-Origin": "*"
                     })

	def parse(self, response):
		if response.status == 200:
			tabela = response.css("table[id=tournamentTable]").extract_first()
			
			tb = pd.read_html(tabela.extract_first())
			print(tb)

			#links_jogos = []
			#for el in links_tb:
			#	if "boxscores/" in el and not "index.fcgi" in el: 
			#		self.result_file.write(el + " \n")
	
	#def closed(self, reason):
	#	if reason == "finished":
	#		self.result_file.close()		
	#	else: