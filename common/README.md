# 공통 자료

## 이 저장소에서 공유하는 것

- `splits/*_ids.jsonl`: 동일 분할을 복원하기 위한 원문 없는 ID·원본 위치 정보
- `configs/`: 모델·환경·생성 조건 및 채점 기준
- `scripts/`: 원본 생성과 잠정 채점 코드
- `reports/`: 원문 없는 집계 성적. (잠정 결과이며 전문가 검증이 아님)

## 로컬 데이터 복원

AI Hub 이용 승인을 받은 후 라벨링 ZIP 8개(TL/VL의 4개 진료과)를 준비합니다. 저장소 최상위에서 다음을 실행합니다.

```bash
python common/scripts/prepare_local_data.py --zip-root /path/to/aihub_download
```

파일은 Git 추적에서 제외된 `data/private/train.jsonl`, `valid.jsonl`, `test.jsonl`에 생성됩니다. 원본을 덮어쓰지 않습니다.

학교 GPU 작업으로 생성 코드를 실행할 때는 다음 진입점을 사용합니다. 로그인 노드에서 직접 모델 추론을 실행하지 마세요.

```bash
python common/scripts/generate_base_v2.py --split valid --limit 0
```

경로 이관으로 스크립트 해시가 기존 학교 실행과 달라졌습니다. 기존 `base_*.jsonl`에 새 코드를 이어 쓰지 마세요. 원래 실행 결과·manifest는 별도로 보존합니다.

## 기존 생성·채점 자료 전달

12,990개 train 생성 응답, 추가 검토 표시, 질문별 집계, 654개 평가 응답·채점은 GitHub에 없습니다. 파일에 원문이 포함되어 있으므로 이용·공유 조건을 확인한 승인된 경로로 따로 준비합니다. 코드와 ID만 받았다고 이 결과를 새로 생성·채점할 필요는 없습니다.

기존 채점 코드는 저장소 루트의 `work/`에 로컬 파일을 배치하는 구조입니다. `judge_train_common.py`는 `work/base_train_review_ready.jsonl`, 평가 코드는 `work/base_eval/base_eval_combined.jsonl`을 읽습니다. 새 유료 채점 전에 입력과 예산을 확인하고, 기존 결과를 먼저 확보하세요. `--run`이 없으면 유료 API를 호출하지 않습니다.

## DPO 담당자에게

공통 자료는 선호 쌍이 아닙니다. 질문별로 신뢰할 수 있는 chosen/rejected를 구성하고, 미확정·잘림·기술 오류는 확정 라벨로 쓰지 않습니다. 개별 데이터 구성 기준과 실제 사용 수를 `../dpo`에 기록합니다. valid로 설정을 선택하고 test는 최종 비교에 유지합니다.
