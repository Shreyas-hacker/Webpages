def Scraper(url,dark_web=False):
    import requests
    import random
    from bs4 import BeautifulSoup
    import time

    def get_tor_session():
        session = requests.session()
        #Tor uses 9050 port as default socks port
        if dark_web:
            session.proxies = {'http':'socks5h://127.0.0.1:9050','https':'socks5h://127.0.0.1:9050'}
            time.sleep(15)
        return session
    
    #Make a request through Tor Connection
    #IP visible through Tor
    session = get_tor_session()
    print("Getting...",url)
    result = session.get(url).text
    soup = BeautifulSoup(result, features='html.parser')
    text = soup.get_text()

    return text
