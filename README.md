# Data analysis
Beat the Bookies
Description:
- Taking a Machine Learning approach to build a model that can better predict the outcomes of Premier League football matches than an expert.
The model will be used to predict the outcomes of all matches in the 2019-2020 Premier League season (with the intention of being reproduced for future seasons).
The intention is to place a ten pound stake on the team tipped to win each of the 380 matches in the Premier League season by the model, using William Hill betting odds,
then comparing how much profit/loss the model's predictions returned by the end of the season against simple betting strategies such as picking the favourite, the underdog, the home team,
away team or draw every time.

Data Source:
- https://www.football-data.co.uk - match statistics and WH betting odds.
- https://www.fifaindex.com - fifa stats were scraped from here

Type of analysis:
- Investigative, using historical data to map future predictions.

# Stratup the project

The initial setup.

Create virtualenv and install the project:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
  $ make clean install test
```

Check for beatthebookies in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/beatthebookies`
- Then populate it:

```bash
  $ ##   e.g. if group is "{group}" and project_name is "beatthebookies"
  $ git remote add origin git@gitlab.com:{group}/beatthebookies.git
  $ git push -u origin master
  $ git push -u origin --tags
```

Functionnal test with a script:
```bash
  $ cd /tmp
  $ beatthebookies-run
```
# Install
Go to `gitlab.com/{group}/beatthebookies` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:
```bash
  $ git clone gitlab.com/{group}/beatthebookies
  $ cd beatthebookies
  $ pip install -r requirements.txt
  $ make clean install test                # install and test
```
Functionnal test with a script:
```bash
  $ cd /tmp
  $ beatthebookies-run
```

# Continus integration
## Github
Every push of `master` branch will execute `.github/workflows/pythonpackages.yml` docker jobs.
## Gitlab
Every push of `master` branch will execute `.gitlab-ci.yml` docker jobs.
