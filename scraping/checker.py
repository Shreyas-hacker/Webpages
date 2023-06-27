import requests

url = 'http://ip-api.com/json/'
key = requests.get(url)

if "Singapore" in key.text:
    print("Your VPN might not be on!!")
    safe = False
else:
    safe = True

if safe==True:
    import darkwebScrapper
    darkwebScrapper.Scraper()
else:
    print("IP change failed, try again later.")