"""공유트렁크 멀티태스크 MLP (Phase 3, 설계 §6).

R1 은 **단일태스크**(배리어결과 3-class)로 착수 — 도달시간 헤드·λ 는 R3(A-1 결정,
기둥7 스테이징). 구조는 **트렁크/헤드 분리**(``ModuleDict``)로 두어 R3 에서 도달시간 헤드를
``heads`` 에 추가만 하면 멀티태스크로 확장된다(재작성 불요). 3층 웜스타트 이식도 이 구조.

설계 §6 규율: 입력=벡터(1층이 시간압축 → MLP 필연), 얕고 좁게 시작, 표준 정규화 3종
(드롭아웃·weight decay·early stopping). 강도는 R3 에서 TF·표본 따라 튜닝(D-023 placeholder).

인과: early stopping 내부 홀드아웃은 **시간순 마지막 조각**(셔플 없음) — train 폴드 안에서도
미래를 early-stop 기준으로 안 씀(기둥2). 정규화·NaN 드롭·폴드 경계는 harness 소유(D-019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.research.models.base import ProbaModel

# R1 placeholder 하이퍼 (D-023, R3 튜닝). 얕고 좁게(설계 §6).
_DEFAULT_PARAMS = {
    "trunk_width": 32,
    "trunk_depth": 2,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 200,
    "patience": 20,
    "es_frac": 0.15,   # train 폴드 내부 early-stop 홀드아웃 비율(시간순 tail)
    "min_train": 50,    # 이보다 작으면 early stopping 없이 고정 epoch
    "lam": 0.1,         # 멀티태스크 도달시간 손실 가중(reach 제공 시만; R3, 낮게 시작)
}


class _MTLNet(nn.Module):
    """트렁크(공유) + 헤드(ModuleDict). barrier 헤드 + (멀티태스크 시) reach 헤드.

    ``multitask=False`` 면 barrier 헤드만 생성 → 단일태스크가 R1 과 **바이트 동일**(reach
    헤드가 초기화 RNG 를 소비해 barrier 초기값을 흔드는 일 없음, R3 correctness).
    """

    def __init__(self, in_dim: int, n_classes: int, width: int, depth: int,
                 dropout: float, multitask: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        self.trunk = nn.Sequential(*layers)
        heads = {"barrier": nn.Linear(d, n_classes)}
        if multitask:
            heads["reach"] = nn.Linear(d, 1)   # 늦게 분기(트렁크 넉넉히 공유, 설계 §6)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        return {name: head(z) for name, head in self.heads.items()}


class SmallMLP(ProbaModel):
    """배리어결과 MLP. ``reach=None`` 이면 단일태스크(=R1), reach(회귀 타깃 Series) 제공 시
    멀티태스크(co-training). 단일클래스 train 폴드는 상수확률 폴백.

    ``reach`` = X.index 정렬된 도달시간 회귀 타깃(예: first_touch/N). **NaN 샘플은 손실에서
    제외**(검열 마스킹) — expire 제외 등 도메인 결정은 호출부(러너) 소유(모델은 generic).
    reach 예측은 학습 co-training 신호일 뿐 ``predict_proba``(배리어)에 안 쓰임.
    """

    def __init__(self, seed: int = 0, params: dict | None = None,
                 reach: "pd.Series | None" = None) -> None:
        self.seed = seed
        self.p = {**_DEFAULT_PARAMS, **(params or {})}
        self.reach = reach
        self.net_: _MTLNet | None = None
        self.classes_: list = []

    def fit(self, X, y) -> "SmallMLP":
        y = pd.Series(y)
        self.classes_ = sorted(y.unique())
        if len(self.classes_) < 2:
            self.net_ = None                     # 단일클래스 폴드 → 상수확률
            return self

        torch.manual_seed(self.seed)             # 가중치 초기화·드롭아웃 재현성
        multitask = self.reach is not None
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        Xv = torch.tensor(np.asarray(X, dtype=np.float32))
        yv = torch.tensor(y.map(cls_to_idx).to_numpy(), dtype=torch.long)

        # 멀티태스크: 도달시간 회귀 타깃·마스크(NaN=검열 제외; 정규화·expire 마스킹은 러너 소유)
        rt = rm = rt_tr = rm_tr = None
        if multitask:
            r = self.reach.reindex(X.index).to_numpy(dtype=np.float32)
            rm = torch.tensor((~np.isnan(r)).astype(np.float32))
            rt = torch.tensor(np.nan_to_num(r, nan=0.0))

        # 시간순 내부 홀드아웃 (셔플 없음 — 인과)
        n = len(yv)
        n_es = int(n * self.p["es_frac"])
        use_es = n_es > 0 and (n - n_es) >= self.p["min_train"]
        if use_es:
            x_tr, y_tr = Xv[: n - n_es], yv[: n - n_es]
            x_es, y_es = Xv[n - n_es:], yv[n - n_es:]
            if multitask:
                rt_tr, rm_tr = rt[: n - n_es], rm[: n - n_es]
        else:
            x_tr, y_tr, x_es, y_es = Xv, yv, None, None
            if multitask:
                rt_tr, rm_tr = rt, rm

        net = _MTLNet(
            in_dim=Xv.shape[1], n_classes=len(self.classes_),
            width=self.p["trunk_width"], depth=self.p["trunk_depth"],
            dropout=self.p["dropout"], multitask=multitask,
        )
        opt = torch.optim.Adam(
            net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"]
        )
        loss_fn = nn.CrossEntropyLoss()
        lam = float(self.p["lam"])

        best_state, best_loss, bad = None, float("inf"), 0
        for _ in range(self.p["max_epochs"]):
            net.train()
            opt.zero_grad()
            out = net(x_tr)
            loss = loss_fn(out["barrier"], y_tr)   # full-batch (결정적)
            if multitask and lam > 0.0 and float(rm_tr.sum()) > 0.0:
                # 도달시간 MAE — 마스크(검열 제외) 샘플만. 트렁크 co-training 신호(설계 §5).
                reach_pred = out["reach"].squeeze(1)
                loss = loss + lam * (torch.abs(reach_pred - rt_tr) * rm_tr).sum() / rm_tr.sum()
            loss.backward()
            opt.step()
            if not use_es:
                continue
            net.eval()
            with torch.no_grad():
                es_loss = loss_fn(net(x_es)["barrier"], y_es).item()   # 조기종료=배리어만
            if es_loss < best_loss - 1e-6:
                best_loss, bad = es_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.p["patience"]:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        idx = self._index(X)
        if self.net_ is None:
            data = np.ones((len(idx), 1))
            return pd.DataFrame(data, index=idx, columns=self.classes_)
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(np.asarray(X, dtype=np.float32)))["barrier"]
            proba = torch.softmax(logits, dim=1).numpy()
        return pd.DataFrame(proba, index=idx, columns=self.classes_)
