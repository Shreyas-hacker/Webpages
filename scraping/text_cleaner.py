import json
import pandas as pd

# open text file to read it
data_list = []
with open("../darknet.txt", 'r', encoding='utf-8') as darknet:
    #iterate over each line in the file
    for line in darknet:
        data = json.loads(line)
        # need to convert the json string to python dictionary (dk why doing it twice then it works)
        data_dict = json.loads(data)
        if 'summary' not in data_dict:
            data_dict['summary'] = ''
        data_list.append((data_dict['url'],data_dict['title'],data_dict['body_stripped'],data_dict['summary']))

#convert list of tuples to dataframe
df = pd.DataFrame(data_list, columns=['url','title','body_stripped','summary'])
# save the dataframe to csv
df.to_csv('darknet.csv', index=False)
