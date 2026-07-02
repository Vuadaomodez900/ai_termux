# ai_termux_ultimate.py (bổ sung chức năng tự cập nhật code)
# ============================================================
import os, sys, json, time, random, sqlite3, threading, subprocess, re, math, hashlib
from datetime import datetime
from collections import defaultdict
try:
    import requests, numpy as np
except:
    os.system("pip install requests numpy -q")
    import requests, numpy as np

# ============================================================
# CẤU HÌNH
# ============================================================
CFG = {
    "max_mem": 2000, "emb_dim": 128, "db": "knowledge.db", 
    "server": "http://localhost:8080", "interval": 3600,
    "lr": 0.01, "hidden": 12,
    "repo_url": "https://raw.githubusercontent.com/yourusername/ai_termux/main/ai_termux_ultimate.py"
}

# ============================================================
# LỚP NEURAL NET NHỎ (TRAIN TRÊN TERMUX)
# ============================================================
class TinyNN:
    def __init__(self, inp=4, hid=12, out=2):
        self.lr=CFG["lr"]
        self.w1 = [[random.uniform(-1,1) for _ in range(hid)] for _ in range(inp)]
        self.w2 = [[random.uniform(-1,1) for _ in range(out)] for _ in range(hid)]
        self.b1 = [0.0]*hid; self.b2 = [0.0]*out
        self.hid_in = self.hid_out = self.out_in = self.out = None
    def sig(self,x): return 1/(1+math.exp(-x))
    def sig_d(self,x): s=self.sig(x); return s*(1-s)
    def soft(self,x): e=[math.exp(v) for v in x]; t=sum(e); return [v/t for v in e]
    def fwd(self,x):
        self.hid_in = [sum(x[i]*self.w1[i][j] for i in range(len(x)))+self.b1[j] for j in range(len(self.b1))]
        self.hid_out = [self.sig(v) for v in self.hid_in]
        self.out_in = [sum(self.hid_out[j]*self.w2[j][k] for j in range(len(self.hid_out)))+self.b2[k] for k in range(len(self.b2))]
        self.out = self.soft(self.out_in); return self.out
    def train(self,x,y):
        self.fwd(x)
        d_out = [self.out[i]-y[i] for i in range(len(y))]
        d_hid = [sum(d_out[k]*self.w2[j][k] for k in range(len(y))) * self.sig_d(self.hid_in[j]) for j in range(len(self.hid_in))]
        for j in range(len(self.w2)):
            for k in range(len(self.w2[0])): self.w2[j][k] -= self.lr*d_out[k]*self.hid_out[j]
        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])): self.w1[i][j] -= self.lr*d_hid[j]*x[i]
        for k in range(len(self.b2)): self.b2[k] -= self.lr*d_out[k]
        for j in range(len(self.b1)): self.b1[j] -= self.lr*d_hid[j]
        return self.out
    def pred(self,x): return self.fwd(x).index(max(self.out))

# ============================================================
# AUTO INSTALLER (CÀI THIẾU)
# ============================================================
class Installer:
    def __init__(self): self.cache=set()
    def _list(self):
        try:
            r=subprocess.run([sys.executable,'-m','pip','list','--format=json'], capture_output=True, text=True)
            if r.returncode==0:
                for p in json.loads(r.stdout): self.cache.add(p['name'].lower())
        except: pass
    def check(self,mod):
        try: __import__(mod); return True
        except: return False
    def ensure(self,mod):
        if self.check(mod): return True
        pkg = {'PIL':'pillow','cv2':'opencv-python','bs4':'beautifulsoup4','sklearn':'scikit-learn'}.get(mod,mod)
        print(f"[Install] {mod} -> {pkg}")
        try:
            subprocess.run([sys.executable,'-m','pip','install',pkg,'-q'], check=True); return True
        except: return False
    def ensure_many(self, lst): return {m:self.ensure(m) for m in lst}

# ============================================================
# RAG + KNOWLEDGE DB
# ============================================================
class RAG:
    def __init__(self):
        self.conn = sqlite3.connect(CFG["db"], check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS k (id INTEGER PRIMARY KEY, topic TEXT, content TEXT, source TEXT, emb BLOB, score REAL DEFAULT 1)''')
        self.cur.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS k_fts USING fts5(topic, content, content='k', content_rowid='id')''')
        self.conn.commit()
        self.emb_cache={}
    def _emb(self,t):
        if t in self.emb_cache: return self.emb_cache[t]
        try:
            r=requests.post(f"{CFG['server']}/embedding", json={"content":t}, timeout=3)
            vec=r.json()["embedding"][:CFG["emb_dim"]]
        except:
            words=t.lower().split()
            vec=[hash(w)%1000/1000.0 for w in words[:CFG["emb_dim"]]]
            vec += [0.0]*(CFG["emb_dim"]-len(vec))
        self.emb_cache[t]=vec; return vec
    def cos(self,a,b): 
        if not a or not b: return 0
        d=sum(x*y for x,y in zip(a,b)); n1=sum(x*x for x in a)**0.5; n2=sum(y*y for y in b)**0.5
        return d/(n1*n2) if n1 and n2 else 0
    def add(self,topic,content,source="user"):
        emb=json.dumps(self._emb(content))
        self.cur.execute("INSERT INTO k(topic,content,source,emb) VALUES(?,?,?,?)",(topic,content,source,emb))
        self.conn.commit()
        self.cur.execute("INSERT INTO k_fts(rowid,topic,content) SELECT id,topic,content FROM k WHERE id=last_insert_rowid()")
        self.conn.commit()
    def search(self,q,k=3):
        qe=self._emb(q)
        self.cur.execute("SELECT id,topic,content,source,score FROM k WHERE id IN (SELECT rowid FROM k_fts WHERE k_fts MATCH ?) ORDER BY score DESC LIMIT ?",(q,k))
        res=[{"topic":r[1],"content":r[2],"source":r[3]} for r in self.cur.fetchall()]
        self.cur.execute("SELECT id,topic,content,source,emb FROM k WHERE emb IS NOT NULL")
        for r in self.cur.fetchall():
            emb=json.loads(r[4].decode() if isinstance(r[4],bytes) else r[4])
            sim=self.cos(qe,emb)
            if sim>0.3 and len(res)<k:
                res.append({"topic":r[1],"content":r[2],"source":r[3]})
        return res[:k]

# ============================================================
# INTERNET SEARCH (TỰ HỌC TỪ WEB)
# ============================================================
def search_web(q):
    res=[]
    try:
        url=f"https://html.duckduckgo.com/html/?q={q.replace(' ','+')}"
        h={"User-Agent":"Mozilla/5.0"}
        r=requests.get(url, headers=h, timeout=10)
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(r.text,'html.parser')
        for a in soup.select('.result__a')[:3]:
            res.append({"title":a.text.strip(),"link":a.get('href',''),"snippet":a.parent.find_next('.result__snippet').text.strip() if a.parent.find_next('.result__snippet') else ""})
    except: pass
    try:
        import wikipedia
        res.append({"title":f"Wiki:{q}","link":f"https://en.wikipedia.org/wiki/{q.replace(' ','_')}","snippet":wikipedia.summary(q,sentences=2)})
    except: pass
    return res

# ============================================================
# AUTO-UPDATE CODE (TỪ GITHUB HOẶC URL)
# ============================================================
def check_for_updates():
    """Kiểm tra và cập nhật code từ remote repository"""
    try:
        print("[Update] Đang kiểm tra phiên bản mới...")
        # Lấy hash của file hiện tại
        current_hash = hashlib.md5(open(__file__, 'rb').read()).hexdigest()
        
        # Tải file mới từ repository
        url = CFG.get("repo_url", "https://raw.githubusercontent.com/yourusername/ai_termux/main/ai_termux_ultimate.py")
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print("[Update] Không thể tải code mới (HTTP", resp.status_code, ")")
            return False
        
        new_code = resp.text
        new_hash = hashlib.md5(new_code.encode('utf-8')).hexdigest()
        
        if current_hash == new_hash:
            print("[Update] Đã là phiên bản mới nhất.")
            return True
        
        # Lưu backup
        backup_path = f"{__file__}.bak"
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(open(__file__, 'r').read())
            print(f"[Update] Đã backup vào {backup_path}")
        
        # Ghi code mới
        with open(__file__, 'w') as f:
            f.write(new_code)
        print("[Update] ✅ Đã cập nhật code mới!")
        print("[Update] Khởi động lại AI để áp dụng thay đổi.")
        return True
    except Exception as e:
        print(f"[Update] Lỗi: {e}")
        return False

def update_from_paste(code_url=None):
    """Cập nhật code từ URL hoặc paste trực tiếp"""
    if code_url:
        try:
            resp = requests.get(code_url, timeout=10)
            if resp.status_code == 200:
                new_code = resp.text
                backup_path = f"{__file__}.bak"
                with open(backup_path, 'w') as f:
                    f.write(open(__file__, 'r').read())
                with open(__file__, 'w') as f:
                    f.write(new_code)
                print("[Update] ✅ Đã cập nhật từ URL.")
                return True
        except:
            print("[Update] Lỗi tải từ URL.")
            return False
    else:
        # Chế độ paste trực tiếp từ terminal
        print("[Update] Dán code mới vào (kết thúc bằng Ctrl+D trên dòng trống):")
        lines = []
        try:
            while True:
                line = input()
                if line == "":  # Ctrl+D sẽ break
                    break
                lines.append(line)
        except EOFError:
            pass
        if not lines:
            print("[Update] Không có code mới.")
            return False
        new_code = "\n".join(lines)
        backup_path = f"{__file__}.bak"
        with open(backup_path, 'w') as f:
            f.write(open(__file__, 'r').read())
        with open(__file__, 'w') as f:
            f.write(new_code)
        print("[Update] ✅ Đã cập nhật từ terminal paste.")
        return True

# ============================================================
# CORE AI (GỘP TẤT CẢ)
# ============================================================
class CoreAI:
    def __init__(self):
        self.nn = TinyNN()
        self.rag = RAG()
        self.installer = Installer()
        self.memory = []
        self.ver = "M1"
        self.evol = 0
        self.installer.ensure_many(['requests','bs4','numpy'])
        threading.Thread(target=self._auto_loop, daemon=True).start()
    def _auto_loop(self):
        while True:
            time.sleep(CFG["interval"])
            self._learn_from_web(random.choice(["python","AI","hacking","security"]))
    def _learn_from_web(self,topic):
        print(f"[AutoLearn] {topic}")
        for r in search_web(topic)[:2]:
            self.rag.add(topic, f"Title:{r.get('title','')}\n{r.get('snippet','')}", r.get('source','web'))
        self.evol += 1; self.ver = f"M{self.evol+1}"
    def reason(self,q):
        ctx = "\n".join([f"- {r['topic']}: {r['content'][:150]}" for r in self.rag.search(q)])
        try:
            r=requests.post(f"{CFG['server']}/completion", json={"prompt":f"Q:{q}\nContext:{ctx}\nA:","max_tokens":200}, timeout=15)
            ans=r.json().get("content","")
        except:
            if ctx: ans=ctx[:500]
            else: 
                w=search_web(q)
                ans=w[0]['snippet'] if w else "Chưa học, sẽ tìm hiểu."
        self.memory.append((q,ans,1.0))
        if len(self.memory)>CFG["max_mem"]: self.memory=self.memory[-CFG["max_mem"]:]
        if len(self.memory)>10:
            x=[hash(m[0])%1000/1000.0 for m in self.memory[-4:]]
            y=[1.0,0.0] if random.random()>0.5 else [0.0,1.0]
            self.nn.train(x,y)
        return ans
    def chat(self):
        os.system("clear" if os.name=="posix" else "cls")
        print("\n"+"="*50)
        print(f"🤖 AI TERMUX ULTIMATE v{self.ver} | /help")
        print("="*50)
        while True:
            cmd=input("\n> ").strip()
            if not cmd: continue
            if cmd=="/exit": break
            if cmd=="/help":
                print("""/learn <topic> - học từ web
/search <q> - tìm web
/packs - list pack đã cài
/install <pkg> - cài pack
/stats - thống kê
/clear - xóa màn hình
/update - cập nhật code từ GitHub
/update <url> - cập nhật từ URL
/paste - dán code mới từ terminal
/restart - khởi động lại AI""")
                continue
            if cmd.startswith("/learn "):
                t=cmd[7:]
                for r in search_web(t)[:3]:
                    self.rag.add(t, f"{r.get('title','')}\n{r.get('snippet','')}", "web")
                print(f"✅ Đã học '{t}'")
                continue
            if cmd.startswith("/search "):
                for r in search_web(cmd[8:])[:3]:
                    print(f"- {r.get('title','')}\n  {r.get('snippet','')[:120]}...")
                continue
            if cmd=="/packs":
                self.installer._list()
                print("Pack đã cài:", ", ".join(list(self.installer.cache)[:20])+"...")
                continue
            if cmd.startswith("/install "):
                self.installer.ensure(cmd[9:])
                continue
            if cmd=="/stats":
                cnt=self.rag.cur.execute("SELECT COUNT(*) FROM k").fetchone()[0]
                print(f"Ver:{self.ver} | Evol:{self.evol} | Mem:{len(self.memory)} | DB:{cnt}")
                continue
            if cmd=="/clear":
                os.system("clear" if os.name=="posix" else "cls")
                continue
            # === CHỨC NĂNG CẬP NHẬT CODE ===
            if cmd == "/update":
                if check_for_updates():
                    print("[Update] Đã cập nhật. Gõ /restart để khởi động lại.")
                continue
            if cmd.startswith("/update "):
                url = cmd[8:].strip()
                if update_from_paste(url):
                    print("[Update] Đã cập nhật từ URL. Gõ /restart để khởi động lại.")
                continue
            if cmd == "/paste":
                if update_from_paste():
                    print("[Update] Đã cập nhật từ dán. Gõ /restart để khởi động lại.")
                continue
            if cmd == "/restart":
                print("[Restart] Đang khởi động lại AI...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                break
            # Mặc định: suy luận
            print(f"\n🧠 {self.reason(cmd)}")

# ============================================================
# TERMINAL UI (CỬA SỔ TERMINAL ẢO - DÙNG CURSES NẾU CÓ)
# ============================================================
try:
    import curses
    HAS_CURSES = True
except:
    HAS_CURSES = False

def terminal_ui():
    if not HAS_CURSES:
        print("Curses không có, dùng chế độ text thường")
        CoreAI().chat()
        return
    
    def _ui(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.clear()
        ai = CoreAI()
        h,w = stdscr.getmaxyx()
        log = ["[System] AI Terminal Ultimate v"+ai.ver]
        
        while True:
            stdscr.clear()
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(0, 0, f" AI TERMUX ULTIMATE v{ai.ver} | {datetime.now().strftime('%H:%M:%S')} ".ljust(w-1))
            stdscr.attroff(curses.A_REVERSE)
            for i,line in enumerate(log[-h+4:]):
                if i < h-4:
                    stdscr.addstr(2+i, 0, line[:w-1])
            stdscr.addstr(h-2, 0, "> ")
            curses.echo()
            cmd = stdscr.getstr(h-2, 2).decode('utf-8').strip()
            curses.noecho()
            if not cmd: continue
            if cmd=="/exit": break
            if cmd.startswith("/learn "):
                t=cmd[7:]
                for r in search_web(t)[:3]:
                    ai.rag.add(t, f"{r.get('title','')}\n{r.get('snippet','')}", "web")
                log.append(f"[Learn] Đã học '{t}'")
            elif cmd=="/packs":
                ai.installer._list()
                log.append(f"[Packs] {', '.join(list(ai.installer.cache)[:10])}...")
            elif cmd.startswith("/install "):
                ok=ai.installer.ensure(cmd[9:])
                log.append(f"[Install] {cmd[9:]}: {'OK' if ok else 'FAIL'}")
            elif cmd=="/stats":
                cnt=ai.rag.cur.execute("SELECT COUNT(*) FROM k").fetchone()[0]
                log.append(f"[Stats] Ver:{ai.ver} Evol:{ai.evol} Mem:{len(ai.memory)} DB:{cnt}")
            elif cmd=="/clear":
                log.clear()
            elif cmd=="/update":
                if check_for_updates():
                    log.append("[Update] Đã cập nhật. Gõ /restart.")
            elif cmd.startswith("/update "):
                url=cmd[8:].strip()
                if update_from_paste(url):
                    log.append("[Update] Đã cập nhật từ URL. Gõ /restart.")
            elif cmd=="/paste":
                log.append("[Update] Dán code mới (Ctrl+D để kết thúc)...")
                curses.echo(); curses.noecho()
                # Không thể paste trực tiếp trong curses, chuyển sang chế độ text
                log.append("[Update] Dùng lệnh /paste ngoài curses (không --ui)")
            elif cmd=="/restart":
                log.append("[Restart] Khởi động lại...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                break
            else:
                ans=ai.reason(cmd)
                log.append(f"🧠 {ans[:w-4]}")
            if len(log)>500: log=log[-300:]
    curses.wrapper(_ui)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--ui":
        terminal_ui()
    else:
        CoreAI().chat()
