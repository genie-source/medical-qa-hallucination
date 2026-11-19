"""Common provisional annotation only; no R-Tuning/DPO training data construction."""
import argparse
from collections import Counter, defaultdict
import getpass
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'work/base_train_review_ready.jsonl'
PROMPT = ROOT / 'common/configs/judge_prompt.json'
FOLDER = ROOT / 'work/common_train_judging_v1'
MODEL = 'claude-haiku-5-5'
INPUT_RATE, OUTPUT_RATE = .10 / 1_000_000, .50 / 1_000_000
MAX_OUTPUT = 2048


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def key(row):
    return (str(row['record_id']), row['generation_index'])


def append(path, row):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def audit(j):
    flags = []
    label, error = j.get('label'), j.get('error_type')
    if label not in ['정답', '오답', '거절', '검토 필요']:
        flags.append('판정 출력 형식 오류')
    if label == '정답' and (j.get('has_clear_error') or error != '없음'):
        flags.append('정답 라벨과 오류 표시 모순')
    if label == '거절' and (not j.get('explicit_refusal') or error != '명시적거절'):
        flags.append('거절 라벨과 명시적 거절 표시 모순')
    if error == '명시적거절' and (label != '거절' or not j.get('explicit_refusal')):
        flags.append('거절 오류유형과 라벨 모순')
    if label != '검토 필요' and (j.get('reference_needs_review') or j.get('generation_failure')):
        flags.append('검토 사유와 확정 라벨 모순')
    if label == '검토 필요':
        flags.append('채점기 검토 요청')
    return ('검토 필요' if flags else label), flags


def export(rows, saved):
    merged, grouped = [], defaultdict(list)
    for r in rows:
        k = key(r)
        if r.get('truncated') or not r['answer'].strip():
            d = {**r, 'provisional_label': '검토 필요', 'review_reasons': ['잘림 또는 빈 답변: API 채점 제외']}
        elif k in saved:
            d = {**r, **saved[k]}
        else:
            d = {**r, 'provisional_label': '채점 대기', 'review_reasons': []}
        merged.append(d)
        grouped[str(r['record_id'])].append(d)
    for name, values in [('all_answers_provisional.jsonl', merged), ('manual_review_queue.jsonl', [r for r in merged if r['provisional_label']=='검토 필요'])]:
        tmp = FOLDER / (name + '.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values), encoding='utf-8')
        tmp.replace(FOLDER / name)
    summaries=[]
    for rid, values in grouped.items():
        counts=Counter(r['provisional_label'] for r in values)
        summaries.append({'record_id':rid, 'qa_id':values[0].get('qa_id'), 'q_type':values[0]['q_type'], 'response_count':len(values), 'counts':dict(counts), 'all_five_resolved':len(values)==5 and not counts['검토 필요'] and not counts['채점 대기']})
    (FOLDER/'question_summary_provisional.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in summaries), encoding='utf-8')
    print('현재 전체 상태:',dict(Counter(r['provisional_label'] for r in merged)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true',help='실제 유료 API 호출. 없으면 준비 상태만 검사')
    parser.add_argument('--limit',type=int,default=100,help='이번 실행 최대 호출 수. 0이면 남은 전체')
    parser.add_argument('--budget-usd',type=float,default=10.,help='이 실행 폴더 누적 추정 비용 한도')
    args=parser.parse_args()
    if args.limit<0 or args.budget_usd<=0: raise ValueError('limit/budget 확인 필요')
    raw=SOURCE.read_bytes(); rows=read_rows(SOURCE)
    if len(rows)!=12990 or len({key(r) for r in rows})!=len(rows): raise ValueError('원본 개수 또는 중복 확인 필요')
    groups=Counter(str(r['record_id']) for r in rows)
    if len(groups)!=2598 or set(groups.values())!={5}: raise ValueError('질문별 답변 수 확인 필요')
    prompt=json.loads(PROMPT.read_text(encoding='utf-8'))
    config={'model':MODEL,'prompt':prompt,'source_sha256':hashlib.sha256(raw).hexdigest(),'max_tokens':MAX_OUTPUT,'effort':'low','audit_version':'common_consistency_v1','estimated_input_usd_per_million':.10,'estimated_output_usd_per_million':.50}
    config_id=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    FOLDER.mkdir(exist_ok=True)
    config_path=FOLDER/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config: raise ValueError('기존 실행 설정과 다름. 결과 보존 후 새 폴더 필요')
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    saved_rows=read_rows(FOLDER/'api_results.jsonl'); saved={key(r):r for r in saved_rows}
    if len(saved)!=len(saved_rows): raise ValueError('저장된 결과 중복 확인 필요')
    for r in saved_rows:
        if r['judge_config_id']!=config_id: raise ValueError('저장된 결과 설정 불일치')
    attempts=read_rows(FOLDER/'api_attempts.jsonl')
    if {key(r) for r in attempts}-set(saved): raise RuntimeError('이전 요청의 응답 저장 여부가 불확실합니다. 중복 과금 방지를 위해 자동 재시도하지 않습니다. 오류만 공유하세요.')
    pending=[r for r in rows if key(r) not in saved and not r.get('truncated') and r['answer'].strip()]
    if args.limit: pending=pending[:args.limit]
    spent=sum(r['estimated_usd'] for r in saved_rows)
    print(f'모델: {MODEL} / 이번 호출: {len(pending)} / 누적 추정 비용: ${spent:.4f} / 한도: ${args.budget_usd:.2f}',flush=True)
    export(rows,saved)
    if not args.run:
        print('준비 검사만 완료. API 호출 없음.');return
    if not pending: return
    print('실제 질문·참고 답변·생성 답변을 Claude API로 전송합니다. 비용은 추정치이며 계정 청구액과 다를 수 있습니다.',flush=True)
    api_key=getpass.getpass('Claude API 키 (화면·파일에 저장하지 않음): ').strip()
    if not api_key: raise ValueError('API 키 없음')
    try:
        for index,row in enumerate(pending,1):
            body={'model':MODEL,'max_tokens':MAX_OUTPUT,'system':prompt['system'],'messages':[{'role':'user','content':json.dumps({k:row[k] for k in prompt['input_fields']},ensure_ascii=False)}],'output_config':{'effort':'low','format':{'type':'json_schema','schema':prompt['schema']}},'service_tier':'standard_only'}
            reserve=(len(json.dumps(body,ensure_ascii=False).encode())+4096)*INPUT_RATE+MAX_OUTPUT*OUTPUT_RATE
            if spent+reserve>args.budget_usd:
                print('누적 추정 비용 한도 도달. 추가 호출 중단.');break
            append(FOLDER/'api_attempts.jsonl',{'record_id':row['record_id'],'generation_index':row['generation_index'],'judge_config_id':config_id,'status':'request_started'})
            req=urllib.request.Request('https://api.anthropic.com/v1/messages',data=json.dumps(body).encode(),method='POST',headers={'Content-Type':'application/json','x-api-key':api_key,'anthropic-version':'2026-10-21'})
            try:
                with urllib.request.urlopen(req,timeout=180) as response: result=json.load(response)
            except urllib.error.HTTPError as e:
                try:
                    detail = json.loads(e.read().decode('utf-8', errors='replace'))
                    error = detail.get('error', {})
                    safe_type = str(error.get('type', 'unknown')).replace(api_key, '[KEY REDACTED]')[:100]
                    safe_message = str(error.get('message', '상세 메시지 없음')).replace(api_key, '[KEY REDACTED]')[:1800]
                except (ValueError, AttributeError):
                    safe_type, safe_message = 'unknown', '상세 오류를 JSON으로 읽을 수 없습니다.'
                print(f'HTTP 오류 {e.code} / 유형: {safe_type} / 사유: {safe_message}', flush=True)
                if e.code == 400:
                    d = {'record_id': row['record_id'], 'generation_index': row['generation_index'],
                         'judge_config_id': config_id, 'status': 'http_400_rejected', 'judge': {},
                         'provisional_label': '검토 필요', 'review_reasons': ['HTTP 400 요청 거절: 의학적 판정 아님'],
                         'http_error_type': safe_type, 'http_error_message': safe_message, 'estimated_usd': 0}
                    append(FOLDER/'api_results.jsonl', d)
                    saved[key(row)] = d
                print('자동 재시도 없이 중단합니다. 오류 유형과 사유를 공유하세요.', flush=True)
                break
            except (urllib.error.URLError, TimeoutError):
                print('연결 오류/시간 초과. 자동 재시도 없음.');break
            usage=result.get('usage',{});cost=usage.get('input_tokens',0)*INPUT_RATE+usage.get('output_tokens',0)*OUTPUT_RATE;spent+=cost
            text=''.join(b.get('text','') for b in result.get('content',[]) if b.get('type')=='text')
            try:
                j=json.loads(text)
                if result.get('stop_reason')!='end_turn': raise ValueError('Incomplete')
                label,flags=audit(j)
            except (ValueError,AttributeError):
                j={};label='검토 필요';flags=['채점 출력 형식 또는 종료 상태 확인 필요']
            d={'record_id':row['record_id'],'generation_index':row['generation_index'],'judge':j,'raw_judge_text':text,'provisional_label':label,'review_reasons':flags,'judge_config_id':config_id,'model_returned':result.get('model'),'message_id':result.get('id'),'stop_reason':result.get('stop_reason'),'usage':usage,'estimated_usd':cost}
            append(FOLDER/'api_results.jsonl',d);saved[key(row)]=d
            print(f'{index}/{len(pending)} · {label} · 누적 추정 ${spent:.4f}',flush=True)
    finally:
        export(rows,saved)
        print('저장 위치:',FOLDER)
        print('자동 판정은 잠정 결과입니다. 검토 항목·표본 확인 전 학습용 확정 라벨로 사용하지 마세요.')


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('\n중단했습니다. 저장된 결과는 보존됩니다.')
