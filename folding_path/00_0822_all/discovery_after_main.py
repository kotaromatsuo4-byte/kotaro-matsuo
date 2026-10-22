"""Use the two main comparison slots for selected source models when free."""
import subprocess,sys,time
from discovery_transfer import ROOT,BASE,DEADLINE

if __name__=='__main__':
    ledger=ROOT/'main_queue_ledger.jsonl'
    while time.time()<DEADLINE-180:
        if ledger.exists() and ledger.read_text(encoding='utf-8').count('\n')>=40:break
        time.sleep(5)
    if time.time()<DEADLINE-180:
        subprocess.run([sys.executable,str(BASE/'discovery_queue.py'),str(ROOT/'selected_adaptation_queue.json'),'--workers','2'],cwd=BASE,check=True)
