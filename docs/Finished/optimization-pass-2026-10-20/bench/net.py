import time, json, requests
from github import Github
R="MichaelJSr/TTSCardMaker"
t=time.perf_counter(); r=requests.get(f"https://api.github.com/repos/{R}/git/trees/main?recursive=1", timeout=30); t1=time.perf_counter()
print("tree GET", r.status_code, f"{(t1-t)*1e3:.0f} ms", len(r.content)/1e6, "MB", "remaining", r.headers.get("X-RateLimit-Remaining"))
ts=[]
for _ in range(7):
    a=time.perf_counter(); d=r.json(); ts.append(time.perf_counter()-a)
print(f"resp.json() per call {min(ts)*1e3:.1f} ms; items {len(d['tree'])}; x7 calls {sum(ts)*1e3:.0f} ms")
psd=[i for i in d["tree"] if i["path"]=="Creatures/Other/5 Stars/Z!!!!!.psd"][0]
g=Github(per_page=1); repo=g.get_repo(R)
a=time.perf_counter(); c=repo.get_commits(path=psd["path"]); n=c.totalCount; b=time.perf_counter(); first=c[0].commit.committer.date; e=time.perf_counter()
print(f"commit lookup: totalCount {n} {1e3*(b-a):.0f} ms, commits[0] {1e3*(e-b):.0f} ms")
a=time.perf_counter(); rr=requests.get(f"https://api.github.com/repos/{R}/git/blobs/{psd['sha']}", headers={"Accept":"application/vnd.github.raw"}, stream=True, timeout=120)
size=sum(len(ch) for ch in rr.iter_content(1<<20)); b=time.perf_counter()
print(f"blob download {size/1e6:.1f} MB in {b-a:.2f}s = {size/1e6/(b-a):.1f} MB/s; remaining {rr.headers.get('X-RateLimit-Remaining')}")
