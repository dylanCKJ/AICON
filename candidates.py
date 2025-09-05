# candidates.py
# 생성된 후보 choice sets를 inputs/candidates.json 에 저장
# 사용 예:
#   python candidates.py --n_sets 2 --alts 3 --seed 42
#   python candidates.py --preset basic_kt_skt --out inputs/candidates.json

import json, os, uuid, random
import argparse
from typing import Dict, List, Any

# === 실험 속성/레벨 & 가격 ===
ATTRS: Dict[str, List[Any]] = {
    "통신사": ["KT", "SKT", "LGU+", "알뜰폰"],
    "데이터": ["10GB", "30GB", "무제한(속도제한)"],
    "통화문자": ["기본제공", "무제한"],
    "5G여부": ["4G", "5G"],
    "테더링": ["5GB", "20GB"],
    "로밍": ["미포함", "월 2GB"],
    "약정": ["무약정", "12개월"],
    "가족결합": ["미적용", "적용"],
    "OTT번들": ["없음", "웨이브", "티빙", "넷플릭스"],
}
PRICE_GRID = [19_000, 29_000, 39_000, 49_000, 59_000]

# === 유틸 ===
def _validate_alt(attrs: Dict[str, Any]) -> None:
    # 각 속성의 값이 정의된 레벨 안에 있는지 검증
    for k, v in attrs.items():
        if k == "가격":
            if v not in PRICE_GRID:
                raise ValueError(f"[가격] {v} 는 price_grid에 없음: {PRICE_GRID}")
            continue
        if k not in ATTRS:
            raise ValueError(f"[속성] '{k}' 가 ATTRS에 없음")
        if v not in ATTRS[k]:
            raise ValueError(f"[레벨] '{k}={v}' 가 정의 밖 값. 허용: {ATTRS[k]}")

def _rand_alt() -> Dict[str, Any]:
    alt = {k: random.choice(vs) for k, vs in ATTRS.items()}
    alt["가격"] = random.choice(PRICE_GRID)
    _validate_alt(alt)
    return alt

def _alts_unique(alts: List[Dict[str, Any]]) -> bool:
    seen = set()
    for a in alts:
        spec = tuple(sorted(a.items()))
        if spec in seen:
            return False
        seen.add(spec)
    return True

def _make_choice_set(keys: List[str], alts: List[Dict[str, Any]], include_no_purchase: bool = True) -> Dict[str, Any]:
    # 키(A,B,C,...)와 대안 속성 매핑
    for attrs in alts:
        _validate_alt(attrs)
    if not _alts_unique(alts):
        raise ValueError("중복된 대안이 있습니다(동일 속성 조합).")
    return {
        "set_id": str(uuid.uuid4())[:8],
        "alternatives": [{"key": k, "attributes": a} for k, a in zip(keys, alts)],
        "include_no_purchase": include_no_purchase,
    }

# === 프리셋(필요할 때 고정된 세트 생성) ===
def preset_basic_kt_skt() -> List[Dict[str, Any]]:
    """
    2개의 고정 세트:
     - 세트1: KT vs SKT (무제한, 부가 조건 다르게)
     - 세트2: LGU+ vs 알뜰폰 (가격/번들 차별)
    """
    set1 = _make_choice_set(
        ["A", "B"],
        [
            {
                "통신사": "KT", "데이터": "무제한(속도제한)", "통화문자": "무제한", "5G여부": "5G",
                "테더링": "20GB", "로밍": "월 2GB", "약정": "12개월", "가족결합": "적용", "OTT번들": "넷플릭스",
                "가격": 49_000,
            },
            {
                "통신사": "SKT", "데이터": "무제한(속도제한)", "통화문자": "기본제공", "5G여부": "5G",
                "테더링": "5GB", "로밍": "미포함", "약정": "무약정", "가족결합": "미적용", "OTT번들": "웨이브",
                "가격": 39_000,
            },
        ],
        include_no_purchase=True,
    )
    set2 = _make_choice_set(
        ["A", "B", "C"],
        [
            {
                "통신사": "LGU+", "데이터": "30GB", "통화문자": "무제한", "5G여부": "4G",
                "테더링": "20GB", "로밍": "월 2GB", "약정": "12개월", "가족결합": "적용", "OTT번들": "티빙",
                "가격": 29_000,
            },
            {
                "통신사": "알뜰폰", "데이터": "10GB", "통화문자": "기본제공", "5G여부": "4G",
                "테더링": "5GB", "로밍": "미포함", "약정": "무약정", "가족결합": "미적용", "OTT번들": "없음",
                "가격": 19_000,
            },
            {
                "통신사": "KT", "데이터": "30GB", "통화문자": "무제한", "5G여부": "5G",
                "테더링": "20GB", "로밍": "월 2GB", "약정": "무약정", "가족결합": "적용", "OTT번들": "웨이브",
                "가격": 39_000,
            },
        ],
        include_no_purchase=True,
    )
    return [set1, set2]

PRESETS = {
    "basic_kt_skt": preset_basic_kt_skt,
}

# === 랜덤 생성 ===
def random_sets(n_sets: int, alts_per_set: int, include_no_purchase: bool = True) -> List[Dict[str, Any]]:
    if not (2 <= alts_per_set <= 4):
        raise ValueError("--alts 는 2~4 사이 권장")
    sets = []
    for _ in range(n_sets):
        alts = []
        # 중복 조합 피하면서 샘플링
        while len(alts) < alts_per_set:
            cand = _rand_alt()
            if cand not in alts:
                alts.append(cand)
        keys = [chr(ord("A") + i) for i in range(alts_per_set)]
        sets.append(_make_choice_set(keys, alts, include_no_purchase))
    return sets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="inputs/candidates.json", help="저장 경로(JSON)")
    ap.add_argument("--n_sets", type=int, default=2, help="랜덤 세트 개수")
    ap.add_argument("--alts", type=int, default=2, help="세트당 대안 수(2~4)")
    ap.add_argument("--no_N", action="store_true", help="미구매 옵션 제외")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    ap.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="프리셋 이름(지정 시 랜덤 무시)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    random.seed(args.seed)

    if args.preset:
        sets = PRESETS[args.preset]()
        print(f"[INFO] Generated preset sets: {args.preset} (n={len(sets)})")
    else:
        sets = random_sets(args.n_sets, args.alts, include_no_purchase=(not args.no_N))
        print(f"[INFO] Generated random sets: n={len(sets)}, alts={args.alts}, include_no_purchase={not args.no_N}")

    obj = {"choice_sets": sets}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved → {args.out}")

if __name__ == "__main__":
    main()
