"""Reconstruct shared splits from locally authorized AI Hub label ZIPs; no downloads."""
import argparse
import json
from pathlib import Path
import unicodedata
import zipfile

ROOT=Path(__file__).resolve().parents[2]
def norm(x):return unicodedata.normalize('NFC',str(x)).lstrip('/')
def main():
    p=argparse.ArgumentParser();p.add_argument('--zip-root',type=Path,required=True);args=p.parse_args()
    if not args.zip_root.is_dir():raise ValueError('ZIP 폴더가 없습니다.')
    specs={s:[json.loads(l) for l in (ROOT/f'common/splits/{s}_ids.jsonl').read_text().splitlines() if l.strip()] for s in ['train','valid','test']}
    required={norm(Path(r['source_zip']).name) for a in specs.values() for r in a}
    zips={}
    for f in args.zip_root.rglob('*.zip'):
        name=norm(f.name)
        if name not in required:continue
        if name in zips:raise ValueError(f'동일 ZIP 이름이 여러 위치에 있습니다: {name}')
        zips[name]=f
    if required-set(zips):raise ValueError(f'누락 ZIP: {sorted(required-set(zips))}')
    output=ROOT/'data/private'
    if any((output/f'{s}.jsonl').exists() for s in specs):raise ValueError('기존 분할 파일이 있습니다. 덮어쓰지 않습니다.')
    wanted={}
    for split,rows in specs.items():
        for r in rows:wanted.setdefault(norm(Path(r['source_zip']).name),[]).append((split,r))
    collected={s:{} for s in specs}
    for name,items in wanted.items():
        with zipfile.ZipFile(zips[name]) as z:
            members={}
            for member in z.namelist():
                n=norm(member)
                if n in members:raise ValueError('ZIP 내부 정규화 이름 중복')
                members[n]=member
            for split,r in items:
                member=members.get(norm(r['source_file']))
                if member is None:raise ValueError(f'ZIP 내부 파일 누락: {name} {r["source_file"]}')
                d=json.loads(z.read(member).decode('utf-8-sig'))
                if any(d[k]!=r[k] for k in ['qa_id','domain','q_type']):raise ValueError('원본 ID·유형 불일치')
                if not all(isinstance(d.get(k),str) and d[k].strip() for k in ['question','answer']):raise ValueError('원본 질문·참고 답변 결측')
                collected[split][r['record_id']]={**r,'question':d['question'],'reference':d['answer']}
    output.mkdir(parents=True,exist_ok=True)
    for split,rows in specs.items():
        values=[collected[split][r['record_id']] for r in rows]
        tmp=output/f'{split}.jsonl.tmp';tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf-8');tmp.replace(output/f'{split}.jsonl')
        print(split,len(values))
    print('로컬 복원 완료. 원문 포함 파일은 GitHub에 업로드하지 마세요.')
if __name__=='__main__':main()
