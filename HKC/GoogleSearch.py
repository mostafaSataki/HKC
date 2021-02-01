from selenium import webdriver

import time
from urllib.parse import quote_plus

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select


class Browser:

    def __init__(self, path='chromedriver.exe', initiate=True, implicit_wait_time = 10, explicit_wait_time = 2):
        self.path = path
        self.implicit_wait_time = implicit_wait_time    # http://www.aptuz.com/blog/selenium-implicit-vs-explicit-waits/
        self.explicit_wait_time = explicit_wait_time    # http://www.aptuz.com/blog/selenium-implicit-vs-explicit-waits/
        if initiate:
            self.start()
        return

    def start(self):
        self.driver = webdriver.Chrome(self.path)
        self.driver.implicitly_wait(self.implicit_wait_time)
        return

    def end(self):
        self.driver.quit()
        return

    def go_to_url(self, url, wait_time = None):
        if wait_time is None:
            wait_time = self.explicit_wait_time
        self.driver.get(url)
        print('[*] Fetching results from: {}'.format(url))
        time.sleep(wait_time)
        return

    def get_search_url(self, query, page_num=0, per_page=10, lang='en'):
        query = quote_plus(query)
        url = 'https://www.google.hr/search?q={}&num={}&start={}&nl={}'.format(query, per_page, page_num*per_page, lang)
        return url

    def scrape(self):
        #xpath migth change in future
        links = self.driver.find_elements_by_xpath("//a[@href]") # searches for all links insede h3 tags with class "r"
        results = []
        for elem in links:
          results.append(elem.get_attribute("href"))


        return results

    def search(self, query, page_num=0, per_page=10, lang='en', wait_time = None):
        if wait_time is None:
            wait_time = self.explicit_wait_time
        url = self.get_search_url(query, page_num, per_page, lang)
        self.go_to_url(url, wait_time)
        results = self.scrape()

        print(results)



    def search2(self):

      wait = WebDriverWait(self.driver, 100)

      wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'listing-item__title')))
      wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'listing-item__price')))

      for elm in self.driver.find_elements_by_css_selector(".listing-item__title,.listing-item__price"):
        print(elm.text)
