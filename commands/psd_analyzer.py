from github import Github
import requests

from commands.query_card import repository

repo = Github().get_repo(repository)

def traverse_repo_files():
    resp = requests.get(f"https://api.github.com/repos/{repository}/git/trees/main?recursive=1")
    if resp.status_code != 200:
        return resp.status_code

    for i in resp.json()["tree"]:
        if i["path"].endswith(".psd"):
            filepath = i["path"]
            print(filepath)
            commits = repo.get_commits(path=filepath)
            date = commits[0].commit.committer.date
            print(date)
