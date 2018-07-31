import scrapy

import re
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from time import gmtime, strftime
import pandas as pd

class NBALinksGames(scrapy.Spider):
	name = 'nba_links_games'

	handle_httpstatus_list = [500]

	anos = [ str(i) for i in range(2018, 2015, -1)]
	meses = ["october", "november", "december", "january", "februray", "march", "april", "may", "june"]

	#start_urls = [
	#   "https://www.basketball-reference.com/leagues/NBA_2018_games-october.html"
	#]

	base_url = "https://www.basketball-reference.com/leagues/NBA_"

	result_file = open('links_jogos.txt', 'a')

	def start_requests(self):	
		for ano in self.anos:
			for mes in self.meses:
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