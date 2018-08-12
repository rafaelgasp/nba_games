import scrapy

import re
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from time import gmtime, strftime
import time
import datetime
import pandas as pd

class NBALinksGames(scrapy.Spider):
	name = 'nba_odds_results'

	handle_httpstatus_list = [500]

	anos = ["2017-2018", "2016-2017", "2015-2016" ]

	de_para_siglas = pd.read_csv('de_para_siglas.csv').set_index('nome').to_dict()['sigla']

	current_dir = "C:/Users/rafae/Google Drive/Projetos/NBA_Games/"

	#base_url = "http://www.oddsportal.com/basketball/usa/nba-2017-2018/results/page/1/"
	base_url = "http://www.oddsportal.com/basketball/usa/nba-"

	df = pd.DataFrame(columns=["game", "home_team", "home_team_odd", "away_team", ])

	def start_requests(self):	
		for ano in self.anos:
			for  in self.meses:
				yield scrapy.Request(self.base_url+ano+"_games-"+mes+".html", callback=self.parse, dont_filter=True)

	def parse(self, response):
		if response.status == 200:
			tbSchedule = response.css("table[id=schedule]")
			links_tb = tbSchedule.css("table[id=schedule]").css("a::attr(href)").extract()
			
			links_jogos = []
			for el in links_tb:
				if "boxscores/" in el and not "index.fcgi" in el: 
					self.result_file.write(el + " \n")
	
	def closed(self, reason):
		if reason == "finished":
			self.result_file.close()		
	#	else: