import scrapy

import re
import os
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from time import gmtime, strftime
from datetime import datetime
import pandas as pd

class NBAGameInfo(scrapy.Spider):
	name = 'nba_games_info'

	handle_httpstatus_list = [500]

	#start_urls = [
	#   "https://www.basketball-reference.com/leagues/NBA_2018_games-october.html"
	#]

	base_url = "https://www.basketball-reference.com"

	jogos_url = [line.rstrip('\n') for line in open('links_jogos.txt')]

	#jogos_url = [jogos_url[2], jogos_url[4]]

	de_para_siglas = pd.read_csv('de_para_siglas.csv').set_index('nome').to_dict()['sigla']

	current_dir = "C:/Users/rafae/Google Drive/Projetos/NBA_Games/"

	def start_requests(self):	
		for url in self.jogos_url:
			yield scrapy.Request(self.base_url + url, callback=self.parse, dont_filter=True)

	def parse(self, response):
		if response.status == 200:
			titulo_pagina = response.css('h1::text').extract_first()

			titulo_pagina = titulo_pagina.replace(",", "")
			titulo_pagina = titulo_pagina.replace("Box Score", "")
			for key in self.de_para_siglas.keys():
				titulo_pagina = titulo_pagina.replace(key, self.de_para_siglas[key])
			titulo_pagina = titulo_pagina.replace(" ", "_")
			titulo_pagina = titulo_pagina.replace("__", "_")
			titulo_pagina = titulo_pagina.replace("at", "@")

			datetime_object = datetime.strptime(titulo_pagina[10:], '%B_%d_%Y')
			titulo_pagina = titulo_pagina[0:10] + datetime_object.strftime("%b_%d_%Y")

			divs_tabelas = response.css('.table_wrapper')
			time_tabela_old = ""
			for i in range(len(divs_tabelas)):
				tabela = divs_tabelas[i].css('table')
				if tabela:
					time_tabela = tabela.css('caption::text').extract_first()
					time_tabela = time_tabela.replace(" Table", "")

					if(len(time_tabela) > 0):
						for key in self.de_para_siglas.keys():
							time_tabela = time_tabela.replace(key, self.de_para_siglas[key])
						time_tabela_old = time_tabela[0:3]

					tb_html = pd.read_html(tabela.extract_first(), header=1)
					df = pd.concat(tb_html)
					#print(df)

					titulo_tabela = tabela.css('.over_header.center::text').extract_first().replace(" Box Score Stats", "")

					arquivo = "dfs/" + titulo_pagina + "_" + time_tabela_old + "_" + titulo_tabela + ".csv"
					df.to_csv(self.current_dir + arquivo)
					print(arquivo)
	
	def closed(self, reason):
		if reason == "finished":
			print("FIM")
			#self.result_file.close()		
	#	else: