# CLAUDE.md

이 파일은 Claude 가 새 세션 시작 또는 compact 후에 즉시 컨텍스트를 파악하기
위한 정보를 담는다. 프로젝트의 정체성, 응답 규칙, 협업 규칙, 세션 시작
체크리스트, 주요 명령을 인라인으로 통합.

---

## 프로젝트 정체성

**CoinBot** — 매매 로직이 분리된 자동매매 봇 뼈대.

- 전략은 `src/strategy/plugins/<name>.py` 파일 1개 + `config/default.yaml` 섹션 1개로 추가됨
- 엔진(`AbstractEngine` / `CoreEngine` / `BacktestEngine`)은 전략의 존재를 모르고도 동작 가능
- backtest / paper / live 모드 모두 같은 `FeeModel` · `RiskManager` · 사이징 공식 사용 (라이브-백테 일관성)
- OKX 무기한 선물 (`BTC/USDT:USDT`) 기반

---

## 응답 규칙

- 모든 응답은 **한글**로 작성한다.
- naming convention 변경 시 반드시 **명시적으로 고지**한다.
- 전략 변경은 **백테스트 검증 후에만** 라이브에 적용한다.
- **확인되지 않은 사실을 단정적으로 말하지 않는다.** 추정·가정·미검증 영역은 그렇게 명시한다.
- **셸 명령 표기는 두 갈래로 분리한다.** 사용자가 직접 실행할 명령(작업 보고서·가이드·안내 메시지)은 **cmd 문법** (`dir`, `type`, `findstr`, `\` 경로, `%VAR%`, `set VAR=...`). Claude 가 세션 내 Bash 도구로 직접 실행할 때는 **bash 문법** (`ls`, `cat`, `grep`, `/` 경로, `$VAR`). 셸 호환에 의존하지 않는 명령(`git`, `python`, `pytest`, `npm` 등)은 양쪽 동일. PowerShell 도구도 보조적으로 사용 가능.

---

## 협업 규칙

다음 패턴은 이 프로젝트의 작업 흐름 표준이다. 새 세션에서도 동일하게 적용한다.

1. **Phase 분할 + 커밋 승인**: 비자명한 작업은 Phase 단위로 분할하고, 각 Phase 종착 시 사용자 승인 후에만 한 묶음 커밋한다. Phase 내부 Step 단위로는 커밋하지 않는다 (Step 결과 임시 보존은 9번 참조). **모든 git commit·push 는 작업 종류·규모와 무관하게 사용자 명시 승인 후에만 수행한다 — 단발 문서·메모리·설정 갱신도 포함**. 사용자가 명시 요청하지 않은 한 자동 커밋 금지.
2. **작업 보고서 즉시 갱신**: Phase 종착 시 `docs/00_Work_Report/<해당 문서>.md` 의 진행 기록표·잠재 이슈 트래커·커밋 ID를 일괄 갱신한다. Phase 내부 Step 진행 중에는 9번에 따라 메모리 파일에 임시 기록.
3. **잠재 이슈 트래커**: 점검 중 발견된 이슈는 ID(I-NNN)로 등록하고 발생 단계·해결 단계·상태를 추적한다.
4. **검증 흐름**: 코드 변경 시 단위 테스트 → 전체 회귀 → 필요시 end-to-end 시연 후 보고.
5. **솔직한 검증 분류**: 코드 변경 결과를 보고할 때, 단위 테스트로만 검증된 항목과 실 시연으로 검증된 항목을 구분하고 미검증 영역은 명시한다.
6. **결정 사항 제시 형식**: 사용자 결정이 필요한 사안은 (가)/(나)/(다) 옵션 + 추천안 + 근거 형태로 제시한다.
7. **임시 변경 즉시 복구**: 시연·테스트용 임시 변경(예: `default.yaml` 의 `active` 임시 활성)은 작업 종료 시 반드시 원상 복구한다.
8. **구조 점검 단계**: 비자명한 작업의 상세 계획 수립 후, 사용자 결정·구현 착수 직전에 다음 4가지를 자체 점검하고 발견 사항을 계획에 반영한다:
   - **DRY**: 동일 패턴이 N개 파일에 중복되지 않는가? helper/추상화로 단일 출처화 가능한가?
   - **캡슐화**: 한 레이어의 관심사(예: 백테/라이브 모드 차이)가 다른 레이어(예: plugin 추론 로직)로 누출되지 않는가?
   - **미래 확장성**: 현재 가정(예: 모든 plugin entry_tf="15m")에 잠긴 "지금만 작동" 코드를 만들지 않는가?
   - **1회용 코드 분리**: 탐사용/측정용 스크립트(profiling, 한 번 돌리고 폐기)는 commit 제외, 결과 텍스트만 작업 보고서에 보존.
9. **Phase 내부 Step 임시 보존**: Phase가 여러 Step으로 분할된 경우, 각 Step 종착 시 결과·핵심 수치를 메모리 파일(`~/.claude/.../memory/phase_<id>_step<n>.md`)에 즉시 기록한다. Phase 종착 시 모든 Step 결과를 작업 보고서 §13에 일괄 통합하고 커밋한 직후 임시 메모리 파일을 삭제한다. 목적: compact/세션 단절 시 컨텍스트 손실 방지. **모든 Step에 일관 적용** — 작업 시간(짧음/긺) 무관, 코드 작성처럼 결과 형태가 비정형이어도 변경 매트릭스 + 검증 결과(pytest/회귀) + 핵심 수치 형태로 짧게라도 작성한다. "짧으니까 면제" 같은 임의 판단 금지 — 사용자가 명시적으로 면제할 때만 예외.
10. **백테 결과 정합성 검증 우선**: 백테 결과 신뢰성 점검 시 데이터 단위 정합성을 먼저 검증한다 — `trades.csv pnl 합` ↔ `metrics.json total_pnl` ↔ `equity_curve.csv 변화량` 일치성. 사용자가 결과를 의심할 때(예: 비현실적 수익률) 직관적 가설(데이터 누출/lookahead)보다 정합성 검증을 우선 수행한다. 불일치 발견 시 즉시 잠재 이슈로 등록. 라이브-백테 PnL 산출 흐름 동등성도 같이 검증한다.

---

## 세션 시작 체크리스트

새 세션 또는 **compact 직후** 다음 순서로 컨텍스트를 파악한다.
(이 문서가 시스템 프롬프트로 자동 주입되지 않은 경우, 사용자에게 명시 요청)

### 1. 작업 보고서 확인

- `docs/00_Work_Report/` 하위 가장 최근 문서를 읽어 진행 상황·잠재 이슈 상태 파악
- 진행 기록 표 마지막 행과 잠재 이슈 트래커의 미해결 항목 확인

### 2. 가이드 확인 (필요 시)

- `docs/01_Guides/USER_GUIDE.md` — 설치·CLI·라이브 운영
- `docs/01_Guides/DEVELOPER_GUIDE.md` — 전략 작성·엔진 hook·신규 모듈 추가

### 3. git 상태

```bash
git status
git log --oneline -10
git diff --stat   # 미커밋 변경이 있다면
```

### 4. config 운영 상태

- `config/default.yaml` 의 `strategies.active` 확인
  - `[]` → 뼈대 상태 (무거래)
  - 비어있지 않음 → 활성 전략 목록 확인
- `mode` 필드는 config에 없음. 실행 모드는 CLI subcommand로 결정 (paper / live / backtest)

### 5. 활성 코드 위치

- `src/core/engine_base.py` — AbstractEngine
- `src/live/engine.py` — CoreEngine (paper/live)
- `src/backtest/engine.py` — BacktestEngine + write_reports
- `src/strategy/plugins/` — 전략 플러그인 폴더 (auto-discovery)
- `src/strategy/base.py` — StrategyModule 추상

---

## 주요 명령

### 백테스트

```bash
python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-12-31
```

- 결과: `data/backtest_reports/00_Working/{tag}_backtest_{start}_{end}_{config_name}/{config_name}/`
- 5종 파일: `trades.csv`, `equity_curve.csv`, `metrics.json`, `config_snapshot.yaml`, `equity_curve.png`

### 페이퍼 / 라이브

```bash
python -m src.main paper --config config/default.yaml
python -m src.main live  --config config/default.yaml
```

### 다중 연도 병렬 백테 + 통합 (Windows)

```bash
scripts\run_full_backtest.bat config/default.yaml
scripts\merge_reports.bat <tag> default
```

### 캔들 다운로드 (수동)

```bash
python scripts/download_history.py --config config/default.yaml --timeframe 1d,4h,15m --start 2020-01-01 --end 2026-04-25
```

### 테스트

```bash
python -m pytest tests/ -q
```

---

## 백테스트 정책

- 백테스트는 기본적으로 **사용자가 직접 수행**한다. 백테스트가 필요하면 커맨드 가이드를 제공할 것.

---

## 잔고 입금 처리 가이드 (BLE-7-3)

사용자가 OKX 잔고 입금 의사를 표명하면 (예: "입금할거야", "잔고 늘릴게") 다음 3 phase 절차를 따른다. 메모 기록 파일: `data/deposits.json` (git untracked).

### Phase 1 — Claude 자동 snapshot (의사 표명 직후)

DB 에서 입금 전 상태 추출 후 사용자에게 입금량·KST 일시·(선택) 메모 확인 요청. 라이브 운영 중이면 graceful shutdown 직전 또는 직후 시점에 수행 (마지막 equity row 가 신뢰할 수 있는 상태).

```bash
python -c "
import sqlite3, json
con = sqlite3.connect('data/coinbot_live.db')
initial = con.execute(\"SELECT value FROM bot_meta WHERE key='initial_balance'\").fetchone()
last_eq = con.execute(\"SELECT timestamp, balance, total_equity FROM equity ORDER BY id DESC LIMIT 1\").fetchone()
peak = con.execute(\"SELECT MAX(total_equity) FROM equity\").fetchone()
print(json.dumps({
    'initial_db': float(initial[0]) if initial else None,
    'last_eq_ts': last_eq[0] if last_eq else None,
    'balance_before': float(last_eq[1]) if last_eq else None,
    'peak_before': float(peak[0]) if peak and peak[0] else None,
}, indent=2))
"
```

### Phase 2 — 사용자 수행 가이드

1. **라이브 graceful shutdown**: `Ctrl+C` (포지션 보유 안전 — 거래소 SL/TP conditional order 유지)
2. **OKX 웹 입금 후 USDT 증가 확인**
3. **라이브 재시작**: `python -m src.main live --config config/ensemble.yaml`
4. **첫 [ACCOUNT] 로그 출력 (~15분 이내) 대기 후** Claude 에게 입금 amount (USDT) + 입금 완료 KST 일시 (+ 선택 메모) 알림

### Phase 3 — Claude 자동 기록·검증 (사용자 알림 후)

1. **DB 신 snapshot 조회 + 검증**

   ```bash
   python -c "
   import sqlite3, json
   con = sqlite3.connect('data/coinbot_live.db')
   last_eq = con.execute(\"SELECT timestamp, balance, total_equity FROM equity ORDER BY id DESC LIMIT 1\").fetchone()
   print(json.dumps({
       'last_eq_ts_after': last_eq[0],
       'balance_after': float(last_eq[1]),
       'total_equity_after': float(last_eq[2]),
   }, indent=2))
   "
   ```

   - `balance_after - balance_before ≈ amount_usdt` 일치 확인 (수수료/오차 ~1 USDT 허용)
   - 불일치 시 사용자에게 alert (입금 일치성 점검 필요)

2. **DB `bot_meta.initial_balance` SQL update** (수익률 기준 재설정)

   ```bash
   python -c "
   import sqlite3
   con = sqlite3.connect('data/coinbot_live.db')
   con.execute(\"UPDATE bot_meta SET value=? WHERE key='initial_balance'\", ('<balance_after>',))
   con.commit()
   print('initial_balance updated to', con.execute(\"SELECT value FROM bot_meta WHERE key='initial_balance'\").fetchone())
   "
   ```

3. **`data/deposits.json` 에 entry append**

   ```bash
   python -c "
   import json, os
   from datetime import datetime, timezone
   path = 'data/deposits.json'
   if os.path.exists(path):
       with open(path) as f: data = json.load(f)
   else:
       data = {'deposits': []}
   entry = {
       'ts_utc': '<KST→UTC 변환된 ISO timestamp>',
       'ts_kst': '<사용자 제공 KST 일시>',
       'amount_usdt': <사용자 제공 amount>,
       'balance_before': <Phase 1 balance_before>,
       'balance_after': <Phase 3 balance_after>,
       'peak_before': <Phase 1 peak_before>,
       'initial_db_before': <Phase 1 initial_db>,
       'initial_db_after': <balance_after>,
       'note': '<사용자 제공 메모 (선택)>',
   }
   data['deposits'].append(entry)
   with open(path, 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
   print('appended:', entry)
   "
   ```

4. **검증 결과 사용자 보고**: balance 증가량 일치 / daily_pnl 영향 0 / peak 자동 갱신 (재시작 시점에 `_restore_state` 의 `update_equity` 가 즉시 갱신) / `data/deposits.json` 누적 entry 표기

### 향후 분석 시 활용

수익률 추적·equity_curve 분석 시 Claude 가 `data/deposits.json` 자동 참조하여 입금 차감 처리. 거래 PnL 만의 추적은 `SELECT SUM(pnl) FROM trades WHERE status='closed'` (입금 무관, 거래 손익만). 입금 시점 이후 PnL 은 `WHERE timestamp > '<최근 입금 ts_utc>'` 조건 추가.

---

## 신규 전략 추가 워크플로 (요약)

상세는 `docs/01_Guides/DEVELOPER_GUIDE.md`.

1. `src/strategy/plugins/my_strategy.py` 작성
   - `@register_strategy` + `StrategyModule` 상속
   - 클래스 속성: `name`, `entry_timeframe`, `required_timeframes`
   - 필수 메서드 3개: `generate_signal` / `compute_stop_loss` / `compute_take_profit`
2. `config/default.yaml` 에 `my_strategy:` 섹션 추가 (필수 키: `risk_per_trade_pct`, `max_leverage`)
3. `strategies.active` 리스트에 `"my_strategy"` 추가

엔진 코드 수정은 0이어야 한다 — 그렇지 않으면 추상화가 잘못된 것.

---

## 폴더 구조 요약

```
src/
├── core/        # AbstractEngine + types/enums/policies/event_bus
├── live/        # CoreEngine (paper/live)
├── backtest/    # BacktestEngine
├── strategy/
│   ├── base.py / registry.py / indicators.py
│   └── plugins/   # ★ 신규 전략
├── execution/   # Broker + OKX/Paper executor
├── risk/        # RiskManager (사이징·DD락·일일한도)
├── accounting/  # FeeModel
├── data/        # feed/historical/store
└── utils/       # logger / config_loader

config/default.yaml
docs/
├── 00_Work_Report/   # 시점별 작업 보고서 (PROTOTYPE_DESIGN_260425.md 등)
└── 01_Guides/        # USER_GUIDE / DEVELOPER_GUIDE
tests/                # pytest
scripts/              # download_history / run_full_backtest / merge_reports
```

---

## 잠재 이슈 트래커

발견된 이슈는 해당 시점 작업 보고서의 "잠재 이슈 트래커" 섹션에 ID(I-NNN)로
등록되어 있다. 새 세션에서도 동일 패턴으로 등록·추적한다.
