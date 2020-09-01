from bs4 import BeautifulSoup
import requests
import pandas as pd


def scrape_historical():
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

    seasons = ['06','07','08','09','10','11','12_8','13_11','14_12','15_16','16_19','17_74','18_174','19_279','20_354']

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

        year = year[:2]
        new_df = pd.DataFrame(data, columns=['Team', 'League', 'ATT', 'MID', 'DEF', 'OVR'])
        new_df['season'] = f'20{str(int(year)-1).zfill(2)}/20{year}'
        df = pd.concat([df, new_df], axis=0)

    df.reset_index(drop=True, inplace=True)
    df.to_csv( "beatthebookies/data/fifarank.csv", index=False, encoding='utf-8-sig')


# create a dictionary to map need all value pairs
team_dict = {'West Bromwich': 'West Brom', 'West Bromwich Albion': 'West Brom',
             'Arsenal FC': 'Arsenal', 'Chelsea FC':'Chelsea', 'Chelsea':'Chelsea',
             'Reading FC': 'Reading', 'Reading':'Reading', 'Arsenal':'Arsenal', 'Bournemouth':'Bournemouth',
             'AFC Bournemouth': 'Bournemouth', 'Sheffield United': 'Sheffield United',
             'Manchester United': 'Man United', 'Liverpool' : 'Liverpool', 'Newcastle United': 'Newcastle',
             'Middlesbrough':'Middlesbrough', 'Bolton Wanderers':'Bolton', 'Everton':'Everton',
             'Hull City': 'Hull', 'Sunderland': 'Sunderland', 'West Ham United': 'West Ham',
             'Aston Villa':'Aston Villa', 'Blackburn Rovers':'Blackburn', 'Fulham':'Fulham',
             'Stoke City':'Stoke', 'Tottenham Hotspur': 'Tottenham', 'Manchester City': 'Man City',
             'Wigan Athletic': 'Wigan', 'Portsmouth':'Portsmouth', 'Wolverhampton Wanderers':'Wolves',
             'Birmingham City': 'Birmingham', 'Burnley': 'Burnley', 'Blackpool':'Blackpool',
             'Queens Park Rangers': 'QPR', 'Swansea City':'Swansea', 'Norwich City': 'Norwich', 'Southampton':'Southampton',
             'Crystal Palace':'Crystal Palace', 'Cardiff City': 'Cardiff', 'Leicester City':'Leicester',
             'Watford':'Watford', 'Brighton & Hove Albion': 'Brighton', 'Huddersfield Town': 'Huddersfield'
            }

def scrape_new():
    url = f"https://www.fifaindex.com/teams/fifa21/?league=13&order=desc"
    while(True):
        print("Getting page for 21")
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
    df['season'] = '2020/2021'
    df.to_csv( "beatthebookies/data/fifarank21.csv", index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    # scrape_historical()
    scrape_new()


