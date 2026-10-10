"""Build a Kaggle kernel folder from kernel_template.py.

    python kaggle_gpu/build.py <config>          # tabpfn | tabpfn-big | tabicl | nn
    kaggle kernels push -p kaggle_gpu/push_<config>

Each config gets its own kernel because the libraries pin conflicting huggingface-hub versions.
Set KAGGLE_USER to your Kaggle username before building."""
import re, sys, os, json
USER = os.environ.get('KAGGLE_USER', 'malikhammadfarooq')
CFG = {
    'tabpfn': dict(pip="(['tabpfn==2.0.9'],)", models="""
    'tabpfn_raw': (RAW, tabpfn(16), 1, 3, False, True),
    'tabpfn_eng': (ENG, tabpfn(16), 1, 3, False, True),
    'tabpfn_raw_og': (RAW, tabpfn(16), 1, 3, True, True),
    'tabpfn_eng_og': (ENG, tabpfn(16), 1, 3, True, True),
"""),
    'tabpfn-big': dict(pip="(['tabpfn==2.0.9'],)", models="""
    'tabpfn_raw_og_big': (RAW, tabpfn(32), 1, 8, True, True),
    'tabpfn_top_og': (TOP, tabpfn(32), 1, 4, True, True),
    'tabpfn_rawx_og': (RAW + ['dew_dep', 'cm_s'], tabpfn(32), 1, 4, True, True),
"""),
    'tabicl': dict(pip="(['tabicl'],)", models="""
    'tabicl_raw': (RAW, tabicl(32), 1, 3, False, True),
    'tabicl_raw_og': (RAW, tabicl(32), 1, 3, True, True),
    'tabicl_eng_og': (ENG, tabicl(32), 1, 3, True, True),
"""),
    'nn': dict(pip="(['pytabkit', 'lightning'],)", models="""
    'realmlp': (ENG, realmlp, 1, 5, False, False),
    'realmlp_og': (ENG, realmlp, 1, 5, True, False),
    'tabm': (ENG, tabm, 1, 5, False, False),
    'tabm_og': (ENG, tabm, 1, 5, True, False),
"""),
}
name = sys.argv[1]; c = CFG[name]
t = open('kaggle_gpu/kernel_template.py', encoding='utf8').read()
t = t.replace("(['tabpfn==2.0.9'], ['pytabkit'])", c['pip'])
t = re.sub(r"M = \{.*?\n\}\n", lambda m: "M = {" + c['models'] + "}\n", t, flags=re.S)
t = t.replace('__ORIG__', open('data/Rainfall.csv', encoding='utf8').read())
compile(t, 'k', 'exec')
d = f'kaggle_gpu/push_{name}'; os.makedirs(d, exist_ok=True)
open(f'{d}/rainfall_gpu.py', 'w', encoding='utf8').write(t)
json.dump({"id": f"{USER}/rainfall-gpu-{name}", "title": f"rainfall-gpu-{name}", "code_file": "rainfall_gpu.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": [], "competition_sources": ["rainfall-probability-prediction-challenge"], "kernel_sources": []},
          open(f'{d}/kernel-metadata.json', 'w'), indent=1)
print('built', d)
