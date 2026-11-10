"""Diagnose available data for training."""
from database.connection import get_engine
from sqlalchemy.orm import Session
from preprocessing.preprocessing import load_catalogue, load_observations, load_gaps, build_grid, select_signals
from feature_engineering.rolling_features import build_features, feature_columns
from config import load_config

cfg = load_config()
eng = get_engine()
with Session(eng) as s:
    cat = load_catalogue(s, 'OKUMA.123456')
    obs = load_observations(s, 'OKUMA.123456')
    gaps = load_gaps(s, 'OKUMA.123456', obs)
    print(f'obs shape: {obs.shape}')
    print(f'gaps: {len(gaps)}')

    sigs = select_signals(cat, obs, cfg)
    print(f'signals selected: {len(sigs)}: {sigs[:10]}')

    if sigs:
        grid = build_grid(obs, cat, gaps, sigs, cfg)
        print(f'grid shape: {grid.shape}')
        if not grid.empty:
            print('state distribution:')
            print(grid['__state'].value_counts())
            print('segment sizes:')
            seg_sizes = grid.groupby('__segment').size()
            print(seg_sizes)
            print(f'\nTotal rows per state:')
            for st in ['active', 'idle', 'unknown']:
                n = (grid['__state'] == st).sum()
                print(f'  {st}: {n} rows')

            feats = build_features(grid, sigs, cfg)
            cols = feature_columns(feats)
            print(f'\nfeature rows: {len(feats)}, feature cols: {len(cols)}')
            if not feats.empty:
                print('feature state distribution:')
                print(feats['__state'].value_counts())
                for st in ['active', 'idle']:
                    sub = feats[feats['__state'] == st]
                    usable = [c for c in cols if sub[c].notna().mean() >= 0.8 and sub[c].nunique() > 1]
                    import pandas as pd
                    X = sub[usable].dropna(thresh=int(0.8 * len(usable))) if usable else sub.iloc[0:0]
                    print(f'  {st}: {len(X)} usable rows, {len(usable)} usable features')
        else:
            print('Grid is EMPTY - check min_segment_rows config')
    else:
        print('No signals selected!')
        # Show why
        cand = cat[(cat.category == 'SAMPLE') & (cat.representation == 'VALUE')].id.tolist()
        print(f'SAMPLE candidates: {len(cand)}')
        for did in cand[:10]:
            v = obs.loc[obs.data_item_id == did, 'value_num'].dropna()
            print(f'  {did}: {len(v)} non-null, {v.nunique()} unique')
