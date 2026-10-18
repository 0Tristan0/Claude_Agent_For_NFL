import json
T=["LAR","KC","CAR","BAL","DET","CHI","BUF","NE","SF","ATL","DAL","CIN","HOU","LV","JAX","GB","SEA","WAS","PIT","NO","NYG","TB","ARI","IND","LAC","NYJ","MIN","MIA","CLE","DEN","PHI","TEN"]
off_tot="""LAR 21.2 126.8 290.2 417.0
KC 29.5 162.2 249.8 412.0
CAR 30.2 104.5 301.2 405.8
BAL 29.0 153.5 241.5 395.0
DET 32.8 114.2 279.6 393.8
CHI 23.8 173.2 219.2 392.4
BUF 31.8 139.0 247.2 386.2
NE 22.0 127.2 245.2 372.4
SF 30.8 128.6 236.4 365.0
ATL 24.0 181.0 179.5 360.5
DAL 27.6 81.6 269.6 351.2
CIN 25.6 82.8 260.0 342.8
HOU 22.0 79.0 261.8 340.8
LV 26.4 82.0 253.6 335.6
JAX 24.6 114.2 220.6 334.8
GB 20.8 60.4 271.0 331.4
SEA 26.8 106.0 222.0 328.0
WAS 23.0 139.6 187.8 327.4
PIT 22.2 101.2 223.4 324.6
NO 23.6 81.8 240.2 322.0
NYG 21.8 109.0 199.2 308.2
TB 20.0 133.0 173.4 306.4
ARI 21.6 91.6 209.6 301.2
IND 24.8 100.8 193.0 293.8
LAC 16.8 128.2 164.6 292.8
NYJ 18.6 94.0 190.0 284.0
MIN 21.8 116.0 167.0 283.0
MIA 14.0 126.6 153.6 280.2
CLE 18.6 94.2 179.4 273.6
DEN 19.4 86.2 182.8 269.0
PHI 18.4 91.6 170.6 262.2
TEN 11.6 82.6 163.8 246.4"""
def_tot="""LAR 20.8 102.0 145.5 247.5
SEA 21.0 100.4 150.8 251.2
MIN 12.8 86.6 173.2 259.8
NYJ 19.2 118.0 175.0 293.0
NE 18.8 90.8 205.4 296.2
SF 18.6 97.6 199.6 297.2
TB 22.4 72.8 226.6 299.4
GB 22.4 127.6 181.0 308.6
CHI 19.2 96.0 215.4 311.4
LAC 25.8 118.6 193.0 311.6
HOU 22.4 78.4 234.0 312.4
ATL 23.0 48.2 264.8 313.0
BAL 24.0 100.0 215.2 315.2
KC 19.2 92.8 225.0 317.8
PIT 21.8 103.4 220.8 324.2
NYG 21.2 127.6 199.4 327.0
MIA 26.4 110.6 219.2 329.8
JAX 14.0 63.8 266.2 330.0
TEN 21.8 114.0 217.8 331.8
CLE 22.4 117.4 216.6 334.0
CIN 21.8 123.0 221.2 344.2
DEN 22.2 145.2 202.2 347.4
WAS 29.8 95.4 255.0 350.4
PHI 22.4 118.0 238.8 356.8
NO 30.2 148.8 212.2 361.0
ARI 32.4 126.6 236.8 363.4
BUF 26.8 116.0 256.2 372.2
LV 25.8 140.6 236.4 377.0
IND 27.6 130.2 254.0 384.2
CAR 27.2 156.5 230.8 387.2
DAL 27.2 168.4 227.0 395.4
DET 29.6 107.0 297.4 404.4"""
def_pass="""LAR 4 53.5 4.84 7 3 7 75.4
SEA 5 63.4 5.45 8 7 14 76.0
MIN 5 53.2 6.26 3 6 17 62.9
NYJ 5 62.5 6.11 6 1 17 89.5
GB 5 65.0 6.35 6 7 14 76.8
LAC 5 64.5 7.27 7 4 11 90.9
NYG 5 65.2 6.51 9 6 8 86.7
SF 5 64.4 6.06 5 3 11 83.4
DEN 5 57.6 6.68 7 5 13 79.4
NE 5 66.0 6.87 6 3 11 90.4
NO 5 70.6 7.46 8 1 10 106.7
CHI 5 63.8 8.79 6 3 12 97.7
CLE 5 59.4 6.85 10 4 16 89.7
TEN 5 74.0 7.83 5 2 10 102.1
MIA 5 76.6 7.79 11 3 4 115.0
PIT 5 64.6 7.39 5 4 11 86.7
CIN 5 62.5 7.29 5 3 17 87.0
BAL 5 65.5 6.96 5 3 10 88.1
KC 4 61.2 6.23 4 4 4 76.8
TB 5 66.3 7.07 7 5 6 88.3
DAL 5 72.1 8.28 10 0 5 120.5
CAR 4 61.1 6.74 5 3 8 84.0
HOU 5 67.2 7.24 6 2 18 94.9
LV 5 62.5 7.91 11 4 13 99.6
ARI 5 63.3 9.08 14 3 11 117.3
PHI 5 62.4 6.98 6 3 6 87.3
ATL 5 61.4 6.22 9 1 11 91.5
IND 5 62.9 8.02 9 1 10 103.4
WAS 5 65.3 7.07 14 3 9 103.9
BUF 4 63.9 7.46 10 2 10 103.4
JAX 5 65.2 7.79 4 6 11 82.3
DET 5 68.1 7.54 14 3 20 106.3"""
off_pass="""CAR 4 62.0 7.78 9 2 10 99.4
LAR 4 61.2 7.28 6 6 6 80.4
DET 4 67.7 7.54 9 0 8 108.6
NO 4 67.2 6.68 9 3 12 95.7
CIN 4 71.3 7.46 7 3 9 99.5
DAL 5 67.2 7.19 9 3 6 97.2
HOU 4 61.2 7.51 5 0 12 95.3
SF 4 69.9 8.91 11 1 0 126.2
LV 4 64.6 7.12 11 4 5 99.5
KC 4 64.1 8.10 9 2 6 106.2
GB 4 56.5 6.76 8 3 8 86.5
BUF 4 66.4 8.73 6 3 9 100.1
BAL 4 72.4 9.81 6 1 7 118.4
SEA 4 68.6 8.44 10 3 6 112.1
CHI 4 68.2 7.39 4 2 7 93.5
PIT 4 56.8 6.45 7 4 13 80.6
JAX 5 67.1 8.21 10 3 10 106.8
NE 4 62.6 7.32 4 7 10 71.9
NYJ 4 72.6 7.71 5 0 12 109.0
DEN 4 60.0 6.07 5 3 5 80.4
CLE 4 69.8 7.37 6 2 11 101.0
ARI 4 67.8 5.38 6 4 7 83.2
WAS 4 56.9 5.74 8 1 5 89.9
LAC 4 59.2 6.80 4 6 12 70.0
ATL 4 63.6 7.75 3 6 8 72.2
IND 4 63.6 5.71 3 4 8 73.9
TB 5 65.6 6.30 4 5 17 78.0
NYG 4 61.6 6.51 6 3 11 87.2
TEN 4 63.0 6.10 3 2 6 81.4
PHI 5 57.4 6.02 8 4 13 81.5
MIA 4 56.3 6.91 1 2 11 73.0
MIN 4 61.8 6.45 4 2 13 85.0"""
ypc_o=dict(CHI=4.73,ATL=5.05,BAL=5.07,KC=5.50,WAS=4.72,BUF=5.05,TB=4.89,SF=4.63,LAC=4.36,NE=4.21,LAR=4.83,MIA=4.87,MIN=3.82,DET=4.39,JAX=4.08,NYG=3.71,SEA=3.81,CAR=4.40,PIT=4.69,IND=3.68,CLE=3.74,NYJ=3.59,ARI=3.75,PHI=3.79,DEN=3.62,CIN=3.63,TEN=3.82,LV=2.99,NO=3.53,DAL=3.81,HOU=3.38,GB=3.05)
ypc_d=dict(ATL=2.72,JAX=2.93,TB=3.28,HOU=3.35,KC=3.57,WAS=3.98,MIN=4.15,CIN=4.52,SF=3.73,BAL=3.88,LAR=3.96,SEA=3.64,CHI=4.94,NE=4.38,MIA=3.63,NYG=4.69,PIT=4.34,CLE=4.58,BUF=4.03,ARI=4.20,PHI=3.99,DET=4.44,LAC=3.78,TEN=3.66,NYJ=4.26,IND=4.81,LV=5.17,GB=4.19,DEN=4.76,NO=5.05,CAR=5.40,DAL=5.10)
def parse(s): return {l.split()[0]:[float(x) for x in l.split()[1:]] for l in s.strip().split("\n")}
ot,dt,dp,op=parse(off_tot),parse(def_tot),parse(def_pass),parse(off_pass)
assert len(ot)==len(dt)==len(dp)==len(op)==len(ypc_o)==len(ypc_d)==32
def rank(d,team,high_good=True):
    v=d[team]; vals=list(d.values())
    better=sum(1 for x in vals if (x>v if high_good else x<v))
    ties=sum(1 for x in vals if x==v)
    return better+1, ties>1
stats={}
def add(key,label,d,hg,fmt):
    stats[key]={"label":label,"higher_better":hg}
    for t in ["BUF","LAR"]:
        r,tie=rank(d,t,hg); stats[key][t]={"val":fmt.format(d[t]),"rank":r,"tie":tie}
# offense
add("o_ppg","Points / game",{k:v[0] for k,v in ot.items()},True,"{:.1f}")
add("o_ypg","Total yards / game",{k:v[3] for k,v in ot.items()},True,"{:.1f}")
add("o_pypg","Pass yards / game",{k:v[2] for k,v in ot.items()},True,"{:.1f}")
add("o_ypa","Yards / pass att",{k:v[2] for k,v in op.items()},True,"{:.2f}")
add("o_cmp","Completion %",{k:v[1] for k,v in op.items()},True,"{:.1f}%")
add("o_rate","Passer rating",{k:v[6] for k,v in op.items()},True,"{:.1f}")
add("o_int","INTs thrown / game",{k:v[4]/v[0] for k,v in op.items()},False,"{:.2f}")
add("o_sk","Sacks allowed / game",{k:v[5]/v[0] for k,v in op.items()},False,"{:.2f}")
add("o_rypg","Rush yards / game",{k:v[1] for k,v in ot.items()},True,"{:.1f}")
add("o_ypc","Yards / carry",ypc_o,True,"{:.2f}")
# defense
add("d_ppg","Points allowed / game",{k:v[0] for k,v in dt.items()},False,"{:.1f}")
add("d_ypg","Total yards allowed / game",{k:v[3] for k,v in dt.items()},False,"{:.1f}")
add("d_pypg","Pass yards allowed / game",{k:v[2] for k,v in dt.items()},False,"{:.1f}")
add("d_ypa","Yards / pass att allowed",{k:v[2] for k,v in dp.items()},False,"{:.2f}")
add("d_cmp","Completion % allowed",{k:v[1] for k,v in dp.items()},False,"{:.1f}%")
add("d_rate","Passer rating allowed",{k:v[6] for k,v in dp.items()},False,"{:.1f}")
add("d_ptd","Pass TDs allowed / game",{k:v[3]/v[0] for k,v in dp.items()},False,"{:.2f}")
add("d_int","INTs / game",{k:v[4]/v[0] for k,v in dp.items()},True,"{:.2f}")
add("d_sk","Sacks / game",{k:v[5]/v[0] for k,v in dp.items()},True,"{:.2f}")
add("d_rypg","Rush yards allowed / game",{k:v[1] for k,v in dt.items()},False,"{:.1f}")
add("d_ypc","Yards / carry allowed",ypc_d,False,"{:.2f}")
for k,v in stats.items(): print(f'{v["label"]:28} BUF {v["BUF"]["val"]:>7} #{v["BUF"]["rank"]}{"T" if v["BUF"]["tie"] else ""}   LAR {v["LAR"]["val"]:>7} #{v["LAR"]["rank"]}{"T" if v["LAR"]["tie"] else ""}')
import os; json.dump(stats,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"stats.json"),"w"),indent=1)
