from collections import Counter
from games import require, half_video_blocks, GAME
from xml.etree import ElementTree as ET
raw, meta = require(['atd_raw', 'atd_meta'])
root = ET.parse(meta).getroot()
slots = len(list(root.iter('Player')))
blocks = half_video_blocks(GAME)
count = missing = graded = missing_graded = errors = 0
runs = Counter(); current = 0
prev = [False]*slots
starts = [0]*slots
observations = [0]*slots
active_counts = Counter()
for line in open(raw):
    parts = line.rstrip().split(':')
    if len(parts) != 3: raise ValueError('ATD line malformed')
    fid = int(parts[0]); count += 1
    if fid != count: errors += 1
    frame = fid - 1
    in_half = any(a <= frame <= b for _, a, b in blocks)
    ball_missing = parts[2].startswith('N')
    missing += ball_missing
    if in_half:
        graded += 1; missing_graded += ball_missing
        if ball_missing: current += 1
        elif current: runs[current] += 1; current = 0
    elif current: runs[current] += 1; current = 0
    groups = parts[1].split(';')
    if len(groups) != slots: raise ValueError('slot count mismatch')
    present = [not x.startswith('N') for x in groups]
    if in_half:
        active_counts[sum(present)] += 1
        for i, p in enumerate(present):
            observations[i] += p
            starts[i] += p and not prev[i]
    prev = present if in_half else [False]*slots
if current: runs[current] += 1
print('records',count,'id_order_errors',errors,'slots',slots)
print('ball_missing_all',missing,'ball_missing_graded',missing_graded,'graded',graded)
print('ball_missing_runs',sum(runs.values()),'short_1_25',sum(v for k,v in runs.items() if k<=25),'over_25',sum(v for k,v in runs.items() if k>25),'max_frames',max(runs, default=0))
print('mean_visible_slots_graded',round(sum(k*v for k,v in active_counts.items())/graded,2),'frames_over_22',sum(v for k,v in active_counts.items() if k>22))
print('slot_visible_fragments_graded',sum(starts),'slots_reappearing',sum(x>1 for x in starts))
print('half_blocks',blocks)
