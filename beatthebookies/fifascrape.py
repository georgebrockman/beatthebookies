from bs4 import BeautifulSoup
import requests
import pandas as pd



def scrape():
    url = f"https://www.fifaindex.com/teams/fifa05/?league=13&order=desc"
    while(True):
        print("Getting page for 05")
        try:
          page = requests.get(url)
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print(e)
            continue
        break
    html = page.content
    soup = BeautifulSoup(html,'lxml')

    table = soup.find('table')
    table_rows = table.find_all('tr')

    data = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text for i in td]
        if len(row) > 7:
            data.append(row[1:-1])

    df = pd.DataFrame(data, columns=['Team', 'League', 'ATT', 'MID', 'DEF', 'OVR'])
    df['season'] = '2004/2005'

    seasons = ['06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']

    for year in seasons:
        url = f"https://www.fifaindex.com/teams/fifa{year}/?league=13&order=desc"
        while(True):
            print("Getting page for " + year)
            try:
              page = requests.get(url)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                print(e)
                continue
            break
        html = page.content
        soup = BeautifulSoup(html,'lxml')

        table = soup.find('table')
        table_rows = table.find_all('tr')

        data = []
        for tr in table_rows:
            td = tr.find_all('td')
            row = [i.text for i in td]
            if len(row) > 7:
                data.append(row[1:-1])

        new_df = pd.DataFrame(data, columns=['Team', 'League', 'ATT', 'MID', 'DEF', 'OVR'])
        new_df['season'] = f'20{str(int(year)-1).zfill(2)}/20{year}'
        df = pd.concat([df, new_df], axis=0)

    df.reset_index(drop=True, inplace=True)
    df.to_csv( "beatthebookies/data/fifarank.csv", index=False, encoding='utf-8-sig')



if __name__ == '__main__':
    scrape()


