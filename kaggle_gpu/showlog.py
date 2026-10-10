import json, sys
for f in sys.argv[1:]:
    print('==', f)
    for d in json.load(open(f, encoding='utf8')):
        s = d.get('data', '').rstrip()
        if d.get('stream_name') != 'stderr' or 'Error' in s:
            print(s[:300].encode('ascii', 'replace').decode())
