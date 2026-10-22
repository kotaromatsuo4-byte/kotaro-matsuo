"""Summarize failure modes without treating failure as missing data."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np


def summarize(directory):
    raw=[]
    for file in sorted(directory.glob('seed*_evaluation.csv')):
        with file.open(encoding='utf-8') as stream: raw.extend(csv.DictReader(stream))
    results=[]
    keys=sorted({(r['arm'],int(r['budget']),float(r['goal'])) for r in raw})
    for arm,budget,goal in keys:
        rows=[r for r in raw if (r['arm'],int(r['budget']),float(r['goal']))==(arm,budget,goal)]
        failures=[r for r in rows if not int(r['success'])]
        result=dict(arm=arm,budget=budget,goal=goal,trials=len(rows),failures=len(failures),
                    failure_reasons=str(dict(Counter(r['reason'] for r in failures))))
        for metric in ('flatness','spread_error','strain','steps','projection_calls'):
            result[f'median_{metric}']=np.median([float(r[metric]) for r in rows])
        results.append(result)
    if results:
        with (directory/'failure_summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    for r in summarize(p.parse_args().directory): print(r)
