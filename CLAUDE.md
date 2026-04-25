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

---

## 협업 규칙

다음 패턴은 이 프로젝트의 작업 흐름 표준이다. 새 세션에서도 동일하게 적용한다.

1. **단계 분할 + 커밋 승인**: 비자명한 작업은 단계로 분할하고, 각 단계 완료 시 사용자 승인 후에만 커밋한다.
2. **작업 보고서 즉시 갱신**: 매 단계 완료 시 `docs/00_Work_Report/<해당 문서>.md` 의 진행 기록표·잠재 이슈 트래커·커밋 ID를 즉시 갱신한다.
3. **잠재 이슈 트래커**: 점검 중 발견된 이슈는 ID(I-NNN)로 등록하고 발생 단계·해결 단계·상태를 추적한다.
4. **검증 흐름**: 코드 변경 시 단위 테스트 → 전체 회귀 → 필요시 end-to-end 시연 후 보고.
5. **솔직한 검증 분류**: 코드 변경 결과를 보고할 때, 단위 테스트로만 검증된 항목과 실 시연으로 검증된 항목을 구분하고 미검증 영역은 명시한다.
6. **결정 사항 제시 형식**: 사용자 결정이 필요한 사안은 (가)/(나)/(다) 옵션 + 추천안 + 근거 형태로 제시한다.
7. **임시 변경 즉시 복구**: 시연·테스트용 임시 변경(예: `default.yaml` 의 `active` 임시 활성)은 작업 종료 시 반드시 원상 복구한다.

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
