"""EXAONE base inference only. No training and no external grading API."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from datetime import datetime, timezone

MODEL_ID = 'LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct'
REVISION = 'e949c91dec92095908d34e6b560af77dd0c993f8'


PROMPT_VERSION = 'concise_by_qtype_no_system_v2'
INSTRUCTIONS = {
    2: '다음 의료 질문에 한국어로 답하세요. 핵심 정답만 한 문장으로 작성하세요. 질문을 반복하거나 제목·목록·배경 설명을 덧붙이지 마세요.',
    3: '다음 의료 질문에 한국어로 답하세요. 질문이 요구한 핵심 내용을 3~5문장 이내로 간결하게 작성하세요. 질문을 반복하거나 제목·목록·환자 정보 요약을 덧붙이지 마세요.',
}


def build_prompt(record):
    qtype = int(record['q_type'])
    if qtype not in INSTRUCTIONS:
        raise ValueError('Only short-answer and descriptive questions are supported.')
    return INSTRUCTIONS[qtype] + '\n\n[질문]\n' + record['question']


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                    sort_keys=True).encode()).hexdigest()


def append_json(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='train')
    parser.add_argument('--limit', type=int, default=20,
                        help='Question count; 0 means all questions.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be >= 0')

    root = Path(__file__).resolve().parents[2]
    source = root / 'data' / 'private' / (args.split + '.jsonl')
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    ids = [str(r['record_id']) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate record_id in input')
    for r in records:
        if not isinstance(r.get('question'), str) or not r['question'].strip():
            raise ValueError('Invalid question')

    # Deferred imports allow --help and syntax checks without loading the model.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. Run this script through jobsubmit -G 1.')
    if importlib.metadata.version('transformers') != '4.46.3':
        raise RuntimeError('Expected transformers==4.46.3; do not alter the shared environment silently.')

    output_dir = root / 'results' / 'concise_v2'
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    samples = 5 if args.split == 'train' else 1
    generation = {'do_sample': args.split == 'train', 'max_new_tokens': 512}
    if generation['do_sample']:
        generation.update(temperature=0.7, top_p=0.9, top_k=0)
    config = {
        'model_id': MODEL_ID, 'revision': REVISION,
        'dtype': str(dtype), 'system_prompt': None,
        'input': 'question with concise instruction through apply_chat_template(add_generation_prompt=True)',
        'prompt_version': PROMPT_VERSION, 'instructions_by_q_type': INSTRUCTIONS,
        'generation': generation, 'samples_per_question': samples,
        'seed_scheme': '42 + int(sha256(record_id:generation_index)[:8],16) mod 2**31',
        'data_sha256': source_hash, 'split': args.split,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    config_id = digest(config)
    output = output_dir / f'base_{args.split}.jsonl'
    done = set()
    if output.exists():
        with output.open(encoding='utf-8') as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as e:
                    raise RuntimeError(f'Output line {number} is incomplete; preserve the file and repair it before resuming.') from e
                if row.get('config_id') != config_id:
                    raise RuntimeError('Existing output uses a different config/data/script; use a separate results folder.')
                key = (str(row['record_id']), row['generation_index'])
                if key in done:
                    raise RuntimeError('Duplicate output record; inspect before resuming.')
                done.add(key)

    selected = records if args.limit == 0 else records[:args.limit]
    pending = [(r, j) for r in selected for j in range(samples)
               if (str(r['record_id']), j) not in done]
    print(f'GPU: {torch.cuda.get_device_name(0)} | dtype: {dtype}', flush=True)
    print(f'{args.split}: {len(selected)} questions, {len(pending)} responses pending', flush=True)
    print(f'Output: {output}', flush=True)
    if not pending:
        print('All requested responses already saved.', flush=True)
        return

    manifest = {**config, 'config_id': config_id, 'gpu': torch.cuda.get_device_name(0),
                'packages': {p: importlib.metadata.version(p)
                             for p in ['torch', 'transformers', 'accelerate', 'peft', 'trl']}}
    (output_dir / f'base_{args.split}_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                               trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        torch_dtype=dtype, device_map={'': 0}, attn_implementation='eager')
    model.eval()
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError('Missing EOS token; inspect tokenizer before generation.')
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = next(iter(eos_ids))
    print(f'EOS IDs: {sorted(eos_ids)} | tokenizer EOS: {tokenizer.eos_token_id}', flush=True)
    context_limit = getattr(model.config, 'max_position_embeddings', None)
    if not isinstance(context_limit, int):
        raise RuntimeError('Model context limit unknown; inspect config before generation.')

    with torch.inference_mode():
        for n, (record, j) in enumerate(pending, 1):
            rid = str(record['record_id'])
            seed = (42 + int(hashlib.sha256(f'{rid}:{j}'.encode()).hexdigest()[:8], 16)) % (2**31)
            set_seed(seed)
            # The reference is saved for later grading, never included in model input.
            inputs = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': build_prompt(record)}],
                tokenize=True, add_generation_prompt=True, return_tensors='pt')
            inputs = inputs.to(model.device)
            length = inputs.shape[1]
            if length + generation['max_new_tokens'] > context_limit:
                raise RuntimeError(f'Question exceeds context limit: {rid}. No silent truncation performed.')
            generated = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                **generation, eos_token_id=eos, pad_token_id=pad)
            tokens = generated[0, length:].tolist()
            truncated = (len(tokens) >= generation['max_new_tokens']
                         and (not tokens or tokens[-1] not in eos_ids))
            row = {
                **record, 'split': args.split, 'generation_index': j,
                'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                'seed': seed, 'config_id': config_id, 'model_id': MODEL_ID,
                'model_revision': REVISION, 'prompt_version': PROMPT_VERSION,
                'input_prompt': build_prompt(record),
                'generation_config': generation, 'input_tokens': length,
                'generated_tokens': len(tokens), 'truncated': truncated,
                'created_at': datetime.now(timezone.utc).isoformat(),
            }
            append_json(output, row)
            print(f'[{n}/{len(pending)}] qa_id={record.get("qa_id")} sample={j+1}/{samples} tokens={len(tokens)} truncated={truncated}', flush=True)
    print('Requested generation completed. No training or grading was run.', flush=True)


if __name__ == '__main__':
    main()
