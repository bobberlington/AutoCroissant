from github import Github
from os import path, walk
from os.path import getmtime, isfile
from requests import get

#from commands.query_card import repository
repository = "MichaelJSr/TTSCardMaker"

repo = Github().get_repo(repository)
local_repo = path.expanduser("~/Desktop/TTSCardMaker")
use_local = True

def traverse_repo():
    resp = get(f"https://api.github.com/repos/{repository}/git/trees/main?recursive=1")
    if resp.status_code != 200:
        return resp.status_code

    for i in resp.json()["tree"]:
        if i["path"].endswith(".psd"):
            filepath = i["path"]
            print(filepath)
            commits = repo.get_commits(path=filepath)
            date = commits[0].commit.committer.date
            print(date)

def traverse_local_repo():
    for folder, subs, files in walk(local_repo):
        for file in files:
            print(folder + '/' + file)
            if isfile(folder + '/' + file):
                print(getmtime(folder + '/' + file))
