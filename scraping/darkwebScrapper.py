from bs4 import BeautifulSoup
def Scraper(url):
    import requests
    import random

    def get_tor_session():
        session = requests.session()
        #Tor uses 9050 port as default socks port
        session.proxies = {'http':'socks5h://127.0.0.1:9050','https':'socks5h://127.0.0.1:9050'}
        return session
    
    #Make a request through Tor Connection
    #IP visible through Tor
    session = get_tor_session()
    print("Getting..,",url)
    result = session.get(url).text

    return result

html = Scraper('http://oyioathjz27447gbzevezh3wnqp4l2likld2i22e6ttu5yjtoutmzqid.onion/index.html')
soup = BeautifulSoup(html, features='html.parser')
text = soup.get_text()
print(text)
