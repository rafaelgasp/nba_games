import scrapy

import re

from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from scrapy.shell import inspect_response

from time import gmtime, strftime
import time
import datetime
import pandas as pd

class NBAStatsLinksGames(scrapy.Spider):
	name = 'nbastats_links_games'

	handle_httpstatus_list = [500]

	anos = [ str(i) for i in range(2018, 2015, -1)]
	meses = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]

	start_urls = [
	   "https://stats.nba.com/game/0021700784/scoring/"
	]

	base_url = "https://stats.nba.com/schedule/"

	#https://stats.nba.com/stats/boxscoreusagev2?EndPeriod=10&EndRange=28800&GameID=0021700784&RangeType=0&Season=2017-18&SeasonType=Regular+Season&StartPeriod=1&StartRange=0


	#  REGULAR SEASON
	#
	# temporada 2015-2016: 00215
	# game number (1 - 1230): 01230
	# https://stats.nba.com/game/0021501230/

	# PLAYOFFS 
	#
	# prefixo: 004
	# temporada 2015-2016: 15
	# etapa dos playoffs (01 - 04) [quartas, semi, oitavas, etc.]: 01
	# identificação do número do confronto (0 - 7): 0
	# jogo da série (máx 7): 1
	# https://stats.nba.com/game/0041500101/ 

	#result_file = open('links_jogos ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d %H-%M-%S') + '.txt', 'a')

	def start_requests(self):
		for url in self.start_urls:
			yield scrapy.Request(url, callback=self.parse, dont_filter=True)	
		#for ano in self.anos:
		#	for mes in self.meses:
		

	def parse(self, response):
		inspect_response(response, self)
		if response.status == 200:
			print("OK")
	
	#def closed(self, reason):
		#if reason == "finished":
		#	self.result_file.close()		
	#	else: