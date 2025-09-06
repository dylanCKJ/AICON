from __future__ import annotations

from types import MethodType
import os, re, json, glob, uuid, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smolagents import Tool, ToolCallingAgent, OpenAIServerModel
from duckduckgo_search import DDGS
import pathlib

class InternetSearchTool(Tool):
    name = "InternetSearchTool"
    description = "질문의 의도를 파악하고, 해당 질문을 답하기 위해 검색이 필요한 키워드 뽑아 DuckDuckGo로 검색 스니펫을 모은다. 이 결과를 LLM에게 요약/정리시켜 한국어 답변을 생성한다."
    inputs = {
        "query": {
            "type": "string",
            "description": "검색/질의 문장",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, ddg_max_results: int = 5, search_chars_limit: int = 2500, **kwargs):
        super().__init__(**kwargs)
        self.ddg_max_results = ddg_max_results
        self.search_chars_limit = search_chars_limit
        
        self.organizer = OpenAIServerModel(
            model_id="K-intelligence/Midm-2.0-Base-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="dummy-key",
            max_tokens=512,
            temperature=0
        )

    def _search_internet(self, query: str) -> tuple[List[str], List[str]]:
        """DuckDuckGo에서 스니펫과 URL 목록을 가져온다."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.ddg_max_results))
            filtered = [r for r in results if "body" in r]
            snippets = [r["body"] for r in filtered]
            urls = [
                (r.get("href") or r.get("url") or r.get("link"))
                for r in filtered
                if (r.get("href") or r.get("url") or r.get("link"))
            ]
            return snippets, urls
        except Exception as e:
            print(f"[InternetSearchTool] 검색 오류: {e}")
            return [], []

    def _compose_messages(self, query: str, search_content: str):
        system = {
            "role": "system",
            "content": (
                "너는 한국어로 간결하게 답한다. 아래 검색 스니펫을 참고하되 "
                "사실만 정리하고, 가능하면 출처 URL을 함께 제시하라."
            ),
        }
        user = {
            "role": "user",
            "content": f"질문: {query}\n\n검색 스니펫:\n{search_content}",
        }
        return [system, user]

    def forward(self, query: Optional[str] = None) -> str:
        if not isinstance(query, str) or not query.strip():
            return "[InternetSearchTool] query가 비어있거나 문자열이 아닙니다."

        query = query.strip()
        snippets, urls = self._search_internet(query)
        search_content = "\n".join(snippets)[: self.search_chars_limit]

        messages = self._compose_messages(query, search_content)
        answer = self.organizer(messages)
        return answer.content if hasattr(answer, "content") else str(answer)


class PersonaBuilderAgent:
    """Build persona segments & weights from labeled JSON files (v0.4)."""
    def __init__(self, *, min_quality:float=0.0)->None:
        self.min_quality=min_quality

    def load_dataset(self,path_glob:str)->List[Dict[str,Any]]:
        data=[]
        for fp in glob.glob(path_glob):
            try:
                with open(fp,encoding="utf-8") as f:
                    data.append(json.load(f))
            except Exception as e:
                print("[WARN] parse fail",fp,e)
        return data

    def _extract_personas(self,obj:Dict[str,Any])->List[PersonaRecord]:
        out=[]
        for p in obj.get("info",{}).get("personas",[]):
            if (p.get("evaluation") or {}).get("avg_rating",5)<self.min_quality:
                continue
            profiles=[PersonaProfile(profile=q.get("profile",""), profile_id=q.get("profile_id"), profile_major=q.get("profile_major"), profile_minor=q.get("profile_minor")) for q in (p.get("persona") or [])]
            out.append(PersonaRecord(persona_id=p.get("persona_id",-1), profiles=profiles, eval_avg=(p.get("evaluation") or {}).get("avg_rating")))
        return out

    def _collect_dialogue_summaries(self,obj:Dict[str,Any])->Dict[int,str]:
        m: Dict[int, List[str]] = {}
        for ut in obj.get("utterances", []) or []:
            pid=ut.get("persona_id"); txt=(ut.get("text") or "").strip()
            if not pid or not txt: continue
            m.setdefault(int(pid), []).append(txt)
        summaries: Dict[int,str] = {}
        for pid, texts in m.items():
            sample = " ".join(texts[:4]); tags=[]
            if any(w in sample for w in ["예산","비싸","가성비","미니멀"]): tags.append("예산에 민감")
            if any(w in sample for w in ["운동","헬스","헬스장"]): tags.append("운동 관심 높음")
            if any(w in sample for w in ["빵","베이킹","마들렌","구움과자"]): tags.append("베이킹 선호")
            if any(w in sample for w in ["택배","장비","구매"]): tags.append("구매 빈도 높음")
            summaries[pid] = ", ".join(tags) if tags else sample[:80]
        return summaries

    def derive_traits_from_profiles(self, pr:PersonaRecord)->Dict[str,float]:
        t={"price_sensitivity":0.5,"tech_affinity":0.5,"fitness_focus":0.5,"home_baking":0.5,"brand_loyalty":0.5,"risk_aversion":0.5}
        for pf in pr.profiles:
            s=f"{pf.profile or ''} {pf.profile_minor or ''} {pf.profile_major or ''}"
            if any(k in s for k in ["소심","절약","검소","미니멀"]):
                t["price_sensitivity"]+=0.2; t["risk_aversion"]+=0.1
            if any(k in s for k in ["운동","헬스","피트니스","체육"]):
                t["fitness_focus"]+=0.3
            if any(k in s for k in ["빵","베이킹","쿠키","휘낭시에","마들렌"]):
                t["home_baking"]+=0.3
            if any(k in s for k in ["게이밍","IT","기술","테크"]):
                t["tech_affinity"]+=0.2
            if any(k in s for k in ["단골","브랜드","충성"]):
                t["brand_loyalty"]+=0.2
        return {k: float(max(0.0, min(1.0, v))) for k,v in t.items()}


    def build_segments(self, path_glob: str, *, default_weights: Optional[Dict[str,float]] = None) -> List[PersonaSegment]:
        all_personas: List[PersonaRecord] = []
        for obj in self.load_dataset(path_glob):
            persons = self._extract_personas(obj)
            ds = self._collect_dialogue_summaries(obj)
            for p in persons:
                if p.persona_id in ds:
                    p.dialogue_summary = ds[p.persona_id]
            all_personas.extend(persons)

        # 1) 프로필 → 필터(연령대/성별/직업군 등) 추출 + 보정
        buckets: Dict[str, List[PersonaRecord]] = {}
        raw_counts: Dict[str, int] = {}
        for pr in all_personas:
            fil = {"연령대": None, "성별": None, "직업군": None}
            for pf in pr.profiles:
                if pf.profile_major in fil:
                    fil[pf.profile_major] = pf.profile_minor or pf.profile

            # === fil 채운 직후: 비어 있으면 간이 추론으로 보정 ===
            if not fil["연령대"]:
                for pf in pr.profiles:
                    guess = infer_age(pf.profile or "")
                    if guess: fil["연령대"] = guess; break
            if not fil["성별"]:
                for pf in pr.profiles:
                    guess = infer_gender(pf.profile or "")
                    if guess: fil["성별"] = guess; break

            # === 불완전 축 드롭(옵션) ===
            if DROP_UNKNOWN_AXES and any(fil.get(ax) in (None, "", "미상") for ax in SEGMENT_AXES):
                continue

            # 2) 설정된 축만으로 키 생성
            key = _seg_key_from_axes(fil, axes=SEGMENT_AXES)
            buckets.setdefault(key, []).append(pr)
            raw_counts[key] = raw_counts.get(key, 0) + 1

        # 3) 초기 weight 계산
        weights = {k: len(v) for k, v in buckets.items()}
        tot = sum(weights.values()) or 1
        weights = {k: v / tot for k, v in weights.items()}

        # 4) 작은 버킷 병합 ('기타')
        merged: Dict[str, List[PersonaRecord]] = {}
        merged_counts: Dict[str, int] = {}
        for k, v in buckets.items():
            w = weights.get(k, 0.0)
            target_k = k if w >= MIN_SEGMENT_WEIGHT else MERGE_SMALL_SEGMENT_NAME
            merged.setdefault(target_k, []).extend(v)
            merged_counts[target_k] = merged_counts.get(target_k, 0) + raw_counts.get(k, 0)

        # 5) 병합 후 weight 재정규화
        weights = {k: merged_counts[k] / sum(merged_counts.values()) for k in merged_counts.keys()}

        # 6) PersonaSegment 생성
        segs: List[PersonaSegment] = []
        for k, v in merged.items():
            parts = k.split("/") if k != MERGE_SMALL_SEGMENT_NAME else []
            filt = {}
            for i, ax in enumerate(SEGMENT_AXES):
                filt[ax] = parts[i] if (k != MERGE_SMALL_SEGMENT_NAME and i < len(parts)) else None
            segs.append(PersonaSegment(name=k, personas=v, weight=float(weights[k]), filters=filt))

        # 7) 정규화
        tot = sum(s.weight for s in segs) or 1
        for s in segs: s.weight /= tot

        # 8) traits 집계 (기존 로직 유지)
        for seg in segs:
            if seg.personas:
                vals = [self.derive_traits_from_profiles(p) for p in seg.personas]
                keys = vals[0].keys()
                seg.traits = {kk: float(np.mean([vv[kk] for vv in vals])) for kk in keys}
        return segs


    def fuse_external_evidence(self, segs:List[PersonaSegment], evidence_json:Dict[str,Any], *, c:float=100, m:float=5, beta:float=1.0)->List[PersonaSegment]:
        if not segs: return segs
        w = np.array([max(1e-12, s.weight) for s in segs]); w=w/w.sum()
        eta = c*w
        s_vec = np.zeros(len(segs))
        def exact(seg: PersonaSegment, flt: Dict[str,Any])->bool:
            return all(seg.filters.get(k)==v for k,v in (flt or {}).items() if v is not None)
        for cl in evidence_json.get("claims", []):
            conf = float(cl.get("confidence", 0.0))
            rec = float(np.exp(-float(cl.get("recency_days", 365))/180.0))
            cons = float(np.log(1+int(cl.get("source_count",1)))/np.log(6))
            base = (conf + rec + cons) / 3.0
            signed = base if cl.get("direction", "+") != "-" else -base
            for i, seg in enumerate(segs):
                if exact(seg, cl.get("segment", {})):
                    s_vec[i] += signed
        s_vec = 1/(1+np.exp(-beta*s_vec))
        eta_post = eta + m*s_vec
        w_post = eta_post/eta_post.sum()
        for i, seg in enumerate(segs):
            seg.weight = float(w_post[i])
            if s_vec[i] > 0:
                seg.research_note = (seg.research_note + ", " if seg.research_note else "") + f"s={s_vec[i]:.2f}"
        return segs

class SegmentRun(NamedTuple):
    segment: PersonaSegment
    counts: List[Dict[str,int]]
    result: SimulationResult
    valid_info: List[Dict[str, Any]]  # NEW: [{"set_id":..., "valid":..., "total":..., "rate":...}, ...]

def run_multi_segment(
    sim: PersonaSimulatorAgent,
    segs: List[PersonaSegment],
    sets: List[ChoiceSet],
    *, repeats:int=50, temperature:float=1.0, seed:int=42,
    min_weight: float = 0.01, top_k: Optional[int] = None,
) -> Dict[str, Any]:
    # 0) 세그먼트 필터링/선정
    seg_pool = [s for s in segs if s.weight >= min_weight]
    if not seg_pool:
        seg_pool = sorted(segs, key=lambda x: x.weight, reverse=True)[: min(top_k or 5, len(segs))]
    elif top_k:
        seg_pool = sorted(seg_pool, key=lambda x: x.weight, reverse=True)[:top_k]

    per_segment: List[SegmentRun] = []

    for idx, seg in enumerate(seg_pool):
        ctx = seg.context_snippet()
        counts_all: List[Dict[str,int]] = []
        valid_list: List[Dict[str, Any]] = []  # NEW
        for j, cs in enumerate(sets):
            stats = sim.sample_distribution_stats(
                ctx, cs, seg_traits=seg.traits,
                repeats=repeats, temperature=temperature,
                seed=seed + 1000*idx + j
            )
            counts_all.append(stats["counts"])
            print(f"[DEBUG] seg={seg.name} set={cs.set_id} counts={stats['counts']}")
            total = repeats
            rate = (stats["valid"] / total) if total else 0.0
            valid_list.append({
                "set_id": cs.set_id,
                "valid": stats["valid"],
                "total": total,
                "rate": rate,
            })
        res = sim.fit_mnl_from_sets(sets, counts_all)
        per_segment.append(SegmentRun(segment=seg, counts=counts_all, result=res, valid_info=valid_list))

    # 1) 세트별 시장 가중 Share
    share_acc: Dict[str, Dict[str, float]] = {}
    weight_sum = sum(s.segment.weight for s in per_segment) or 1.0
    for run in per_segment:
        w = run.segment.weight / weight_sum
        if run.result.share_df is None or run.result.share_df.empty:
            continue
        for _, row in run.result.share_df.iterrows():
            sid = row["set_id"]
            share_acc.setdefault(sid, {})
            for col, val in row.items():
                if col == "set_id": 
                    continue
                share_acc[sid][col] = share_acc[sid].get(col, 0.0) + float(val) * w

    # 2) 시장 가중 WTP
    wtp_acc: Dict[str, float] = {}
    wtp_wsum: Dict[str, float] = {}
    for run in per_segment:
        w = run.segment.weight / weight_sum
        if run.result.wtp_df is None or run.result.wtp_df.empty:
            continue
        for _, row in run.result.wtp_df.iterrows():
            f = str(row["feature"]); x = float(row["wtp"])
            if not np.isfinite(x): continue
            wtp_acc[f]  = wtp_acc.get(f, 0.0) + x * w
            wtp_wsum[f] = wtp_wsum.get(f, 0.0) + w
    wtp_market = {f: (wtp_acc[f]/wtp_wsum[f]) for f in wtp_acc.keys() if wtp_wsum.get(f,0)>0}

    # 3) 세트별 valid rate: (a) 세그먼트별, (b) 시장 가중 평균
    valid_by_segment: List[Dict[str, Any]] = []
    for run in per_segment:
        valid_by_segment.append({
            "segment": run.segment.name,
            "weight": run.segment.weight,
            "by_set": {vi["set_id"]: {"valid": vi["valid"], "total": vi["total"], "rate": vi["rate"]} for vi in run.valid_info}
        })

    valid_weighted: Dict[str, float] = {}  # set_id -> weighted valid rate
    for run in per_segment:
        w = run.segment.weight / weight_sum
        for vi in run.valid_info:
            sid = vi["set_id"]
            valid_weighted[sid] = valid_weighted.get(sid, 0.0) + vi["rate"] * w

    aggregate = {
        "market_shares": share_acc,
        "market_wtp": wtp_market,
        "segments_used": [{"name": r.segment.name, "weight": r.segment.weight, "n_personas": len(r.segment.personas)} for r in per_segment],
        "valid_rates": {
            "by_segment": valid_by_segment, # 각 세그먼트-세트별 valid/total/rate
            "weighted_by_market": valid_weighted  # 세트별 시장 가중 평균 valid rate
        }
    }
    return {"per_segment": per_segment, "aggregate": aggregate}

SEGMENT_AXES = ["연령대", "성별"] # 필요 시 ["연령대","성별","직업군"]
DROP_UNKNOWN_AXES = True  # 축 값 없으면 버킷 제외
MIN_SEGMENT_WEIGHT = 0.02 # 2% 미만 → '기타'로 병합
MERGE_SMALL_SEGMENT_NAME = "기타"

def _seg_key_from_axes(filters: Dict[str, Any], axes=SEGMENT_AXES) -> str:
    parts = []
    for ax in axes:
        v = filters.get(ax)
        parts.append(v if v else "미상")
    return "/".join(parts)

# === very light heuristics to infer age/gender from profile text ===
def infer_age(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"(10대|20대|30대|40대|50대|60대 이상|60대|70대 이상)", text)
    if m:
        val = m.group(1)
        return "60대 이상" if ("60대" in val and "이상" in val) or ("70" in val) else val
    if "10대 초반" in text or "10대 이하" in text: return "10대 이하"
    return None

def infer_gender(text: str) -> Optional[str]:
    if not text: return None
    if ("여자" in text) or ("여성" in text): return "여"
    if ("남자" in text) or ("남성" in text): return "남"
    return None

@dataclass
class PersonaProfile:
    profile: str
    profile_id: Optional[int]=None
    profile_major: Optional[str]=None
    profile_minor: Optional[str]=None

@dataclass
class PersonaRecord:
    persona_id: int
    profiles: List[PersonaProfile]
    eval_avg: Optional[float]=None
    eval_grade: Optional[str]=None
    dialogue_summary: Optional[str]=None
    def to_context_text(self)->str:
        return "\n".join([f"- {p.profile_major}: {p.profile}" if p.profile_major else f"- {p.profile}" for p in self.profiles])

@dataclass
class PersonaSegment:
    name: str
    personas: List[PersonaRecord]
    weight: float
    filters: Dict[str,Any]=field(default_factory=dict)
    research_note: Optional[str]=None
    traits: Dict[str,float]=field(default_factory=dict)
    def context_snippet(self, max_personas:int=3)->str:
        subset=self.personas[:max_personas]
        joined="\n\n".join([p.to_context_text()+(f"\n(대화요약) {p.dialogue_summary}" if p.dialogue_summary else "") for p in subset])
        trait_line=""
        if self.traits:
            ks=[
                f"가격민감도:{self.traits.get('price_sensitivity',0.5):.2f}",
                f"기술관심:{self.traits.get('tech_affinity',0.5):.2f}",
                f"운동관심:{self.traits.get('fitness_focus',0.5):.2f}",
                f"브랜드충성:{self.traits.get('brand_loyalty',0.5):.2f}",
            ]
            trait_line="\n[특성] "+", ".join(ks)
        note=f"\n(검색 근거) {self.research_note}" if self.research_note else ""
        return f"[세그먼트: {self.name}]\n{joined}{trait_line}{note}"

@dataclass
class ChoiceAlternative:
    key: str
    attributes: Dict[str,Any]
    def render(self)->str:
        return f"옵션 {self.key}:\n"+"\n".join([f"  - {k}: {v}" for k,v in self.attributes.items()])

@dataclass
class ChoiceSet:
    set_id: str
    alternatives: List[ChoiceAlternative]
    include_no_purchase: bool=True
    def render(self)->str:
        out=[alt.render() for alt in self.alternatives]
        if self.include_no_purchase: out.append("옵션 N: 구매하지 않음")
        return "\n\n".join(out)

@dataclass
class SimulationResult:
    design_matrix: pd.DataFrame
    y_choices: np.ndarray
    coef_df: Optional[pd.DataFrame]=None
    wtp_df: Optional[pd.DataFrame]=None
    share_df: Optional[pd.DataFrame]=None

PROMPT_SYSTEM_KO = (
    "당신은 한 소비자입니다. 당신은 이동통신 요금제를 둘러보던 중 무작위로 설문에 초대되었습니다. "
    "면접원은 당신이 매장에서 보았던 선택지들을 설명하고, 실제로 어떤 선택을 했는지 묻습니다. "
    "항상 두 가지 이상의 옵션이 제시되며, 언제든지 '그날은 아무 것도 구매하지 않음'을 선택할 수도 있습니다. "
    "오직 하나의 JSON 객체만 반환하세요. 필드는 'choice' (값: 후보 키 중 하나 또는 'N')와 "
    "'reason' (한국어 한 줄 사유)입니다. 쌍따옴표를 사용하는 단일 JSON 한 줄만 출력하세요."
    ""
)

@dataclass
class _GenConfig:
    def __init__(self, num_alternatives=2, include_no_purchase=True, price_scale=1_000_000.0):
        self.num_alternatives = num_alternatives
        self.include_no_purchase = include_no_purchase
        self.price_scale = price_scale

class PersonaSimulatorAgent:
    def __init__(self, *, attributes:Dict[str,List[Any]], price_grid:List[float], num_alternatives:int=2, include_no_purchase:bool=True, price_scale:float=1_000_000.0, organizer=None):
        self.organizer = organizer
        self.attributes=attributes
        self.price_grid=price_grid
        self.cfg=_GenConfig(num_alternatives=num_alternatives, include_no_purchase=include_no_purchase, price_scale=float(price_scale))
        self.baseline_levels={k:v[0] for k,v in attributes.items()}

    def _random_alt(self)->ChoiceAlternative:
        attrs={k:random.choice(v) for k,v in self.attributes.items()}
        attrs["가격"]=random.choice(self.price_grid)
        return ChoiceAlternative(key="A", attributes=attrs)

    def generate_choice_sets(self, n_sets:int)->List[ChoiceSet]:
        sets: List[ChoiceSet] = []
        for _ in range(n_sets):
            alts: List[ChoiceAlternative] = []
            seen=set()
            while len(alts)<self.cfg.num_alternatives:
                alt=self._random_alt(); spec=tuple(sorted(alt.attributes.items()))
                if spec in seen: continue
                seen.add(spec); alt.key=chr(ord('A')+len(alts)); alts.append(alt)
            sets.append(ChoiceSet(set_id=str(uuid.uuid4())[:8], alternatives=alts, include_no_purchase=self.cfg.include_no_purchase))
        return sets

    @staticmethod
    def _choice_prompt(persona_ctx: str, choice_set: "ChoiceSet", seg_traits: Optional[Dict[str, float]] = None) -> str:
        keys = [alt.key for alt in choice_set.alternatives]
        if choice_set.include_no_purchase:
            keys.append("N")
        allowed_str = "/".join(keys) # e.g., "A/B/C/N"

        trait_block = ""
        if seg_traits:
            trait_block = (
                "[페르소나 특성]\n"
                f"- 가격민감도: {seg_traits.get('price_sensitivity', 0.5):.2f}\n"
                f"- 기술관심: {seg_traits.get('tech_affinity', 0.5):.2f}\n"
                f"- 운동관심: {seg_traits.get('fitness_focus', 0.5):.2f}\n"
                f"- 브랜드충성: {seg_traits.get('brand_loyalty', 0.5):.2f}\n\n"
            )

        set_block = choice_set.render()
        no_purchase_line = "\n옵션 N: 구매하지 않음" if choice_set.include_no_purchase else ""
        if choice_set.include_no_purchase and "옵션 N:" not in set_block:
            set_block = set_block + no_purchase_line

        return (
            "당신은 아래 페르소나를 가진 한국인 소비자입니다. 옵션 중 하나만 고르세요. "
            f"N=구매하지 않음.\n\n{trait_block}[페르소나]\n{persona_ctx}\n\n[선택 세트]\n{set_block}\n\n"
            f"JSON만: " + "{\"choice\":\"" + allowed_str + "\",\"reason\":\"...\"} (단일 JSON 한 줄)"
        )


    def _sample_once(self, persona_ctx, choice_set, *, temperature=1.0, seed=None, seg_traits=None)->Optional[str]:
        msgs = [
            {"role": "system", "content": PROMPT_SYSTEM_KO},
            {"role": "user", "content": self._choice_prompt(persona_ctx, choice_set, seg_traits)}
        ]
        return self.organizer(msgs, temperature=temperature, seed=seed, max_tokens=512).content


    @staticmethod
    def _parse_choice_json(txt:str)->Optional[Dict[str,Any]]:
        try:
            m=re.search(r"\{[\s\S]*\}", txt)
            if not m:
                return None
            obj=json.loads(m.group(0))
            return obj if isinstance(obj,dict) and "choice" in obj else None
        except Exception:
            return None

    @staticmethod
    def _parse_choice_backup(txt:str)->Optional[str]:
        m=re.search(r'"choice"\s*:\s*"([ABN])"',txt)
        return m.group(1) if m else None

    def sample_distribution(self, *args, **kwargs) -> Dict[str,int]:
        stats = self.sample_distribution_stats(*args, **kwargs)
        total = stats["valid"] + stats["invalid"]
        print(f"[LLM] valid={stats['valid']}/{total} ({(stats['valid']/total if total else 0):.1%}), invalid={stats['invalid']}")
        return stats["counts"]

    def sample_distribution_stats(
        self, persona_ctx: str, choice_set: "ChoiceSet", *,
        seg_traits=None, repeats: int = 30, temperature: float = 1.0, seed: int = 0,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        엄격 모드: 유효 JSON만 카운트. 실패분(형식 오류/허용 외 choice)은 버림.
        반환: {"counts": {...}, "valid": int, "invalid": int, "invalid_examples": [...]}
        """
        counts = {alt.key: 0 for alt in choice_set.alternatives}
        if choice_set.include_no_purchase:
            counts["N"] = 0

        allowed = [alt.key.upper() for alt in choice_set.alternatives]
        if choice_set.include_no_purchase:
            allowed.append("N")
        allowed_set = set(allowed)

        valid = 0
        invalid_examples = []

        for i in range(repeats):
            parsed = None
            for attempt in range(max_attempts):
                txt = self._sample_once(
                    persona_ctx, choice_set,
                    temperature=temperature,
                    seed=seed + i * 100 + attempt,
                    seg_traits=seg_traits
                ) or ""
                obj = self._parse_choice_json(txt)
                if not obj:
                    if attempt == max_attempts - 1:
                        invalid_examples.append(txt.strip()[:200])
                    continue
                ch = str(obj.get("choice", "")).strip().upper()
                if ch not in allowed_set:
                    if attempt == max_attempts - 1:
                        invalid_examples.append(txt.strip()[:200])
                    continue
                parsed = {"choice": ch, "reason": obj.get("reason", "")}
                break

            if parsed:
                counts[parsed["choice"]] += 1
                valid += 1

        return {
            "counts": counts,
            "valid": valid,
            "invalid": repeats - valid,
            "invalid_examples": invalid_examples[:3],  # 디버깅 꼬리 3개만
        }
        

    def _design_matrix(self, cs:ChoiceSet)->Tuple[pd.DataFrame,List[str]]:
        rows: List[Dict[str, Any]] = []
        labels: List[str] = []
        for alt in cs.alternatives:
            row: Dict[str, Any] = {}
            for k,v in alt.attributes.items():
                if k=="가격": row["가격"]=float(v)/self.cfg.price_scale
                else:
                    base=self.baseline_levels.get(k)
                    if base is None: continue
                    if v!=base: row[f"{k}={v}"]=1
            rows.append(row); labels.append(alt.key)
        if cs.include_no_purchase:
            rows.append({"가격":0.0}); labels.append("N")
        return pd.DataFrame(rows).fillna(0.0), labels


    def fit_mnl_from_sets(self, sets: List[ChoiceSet], counts: List[Dict[str, int]]) -> SimulationResult:
        """
        - 정상 케이스: 다항 로짓(MNL) 적합 → coef, WTP, 세트별 share 계산
        - 단일 클래스 등 MNL 불가: coef/WTP는 None, 세트별 share는 관측 빈도 기반 폴백
        """
        X_blocks: List[pd.DataFrame] = []
        y_list: List[int] = []
        feature_bag: set = set()

        for cs, cnt in zip(sets, counts):
            X_cs, labels = self._design_matrix(cs) # (n_alts x p), labels e.g., ["A","B","N"]
            X_cs = X_cs.fillna(0.0)
            feature_bag |= set(X_cs.columns)

            for j, lab in enumerate(labels):
                n = int(cnt.get(lab, 0))
                if n > 0:
                    X_blocks.append(pd.concat([X_cs.iloc[[j]]] * n, ignore_index=True))
                    y_list += [j] * n

        if not X_blocks:
            return SimulationResult(pd.DataFrame(), np.array([]), None, None, None)

        X_all = pd.concat(X_blocks, ignore_index=True).fillna(0.0)
        y_all = np.array(y_list, dtype=int)

        unique_classes = np.unique(y_all)
        if unique_classes.size < 2:
            share_rows: List[Dict[str, Any]] = []
            for cs, cnt in zip(sets, counts):
                total = sum(int(v) for v in cnt.values()) or 1
                row = {"set_id": cs.set_id}
                for lab in ([alt.key for alt in cs.alternatives] + (["N"] if cs.include_no_purchase else [])):
                    row[lab] = float(cnt.get(lab, 0)) / float(total)
                share_rows.append(row)
            share_df = pd.DataFrame(share_rows)
            return SimulationResult(X_all, y_all, coef_df=None, wtp_df=None, share_df=share_df)

        clf = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
        clf.fit(X_all.values, y_all)
        mean_coef = pd.Series(clf.coef_.mean(axis=0), index=X_all.columns)

        # 4) WTP 계산 (가격계수 음수일 때만; 아니면 경고)
        beta_price = mean_coef.get("가격", np.nan)
        wtp_items: List[Dict[str, Any]] = []
        warn_price = False
        if np.isfinite(beta_price) and beta_price < 0:
            for col, b in mean_coef.items():
                if col == "가격":
                    continue
                wtp_items.append({
                    "feature": col,
                    "wtp": float((-b / beta_price) * self.cfg.price_scale)
                })
        else:
            warn_price = True
            print("[WARN] Price coefficient invalid or non-negative; WTP may be unreliable.")

        coef_df = mean_coef.reset_index()
        coef_df.columns = ["feature", "beta_mean"]
        wtp_df = pd.DataFrame(wtp_items)

        # 5) 세트별 예측 점유율 (★ 학습 feature 순서 고정 + reindex 로 보정)
        def softmax(u: np.ndarray) -> np.ndarray:
            e = np.exp(u - u.max())
            return e / e.sum()

        feature_order = list(X_all.columns)  # 학습에 사용된 정확한 열 순서
        share_rows: List[Dict[str, Any]] = []

        for cs in sets:
            X_cs, labels = self._design_matrix(cs)
            X_cs = X_cs.fillna(0.0).reindex(columns=feature_order, fill_value=0.0)
            u = X_cs.values @ mean_coef.values
            p = softmax(u)
            share_rows.append({"set_id": cs.set_id, **{lab: float(pi) for lab, pi in zip(labels, p)}})

        share_df = pd.DataFrame(share_rows)

        return SimulationResult(X_all, y_all, coef_df=coef_df, wtp_df=wtp_df, share_df=share_df)


def load_choice_sets_from_json(path: str) -> List[ChoiceSet]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    out = []
    for cs in obj.get("choice_sets", []):
        alts = [ChoiceAlternative(a["key"], a["attributes"]) for a in cs["alternatives"]]
        sid = cs.get("set_id") or str(uuid.uuid4())[:8]   # ★ 보강
        out.append(ChoiceSet(set_id=sid, alternatives=alts,
                             include_no_purchase=bool(cs.get("include_no_purchase", True))))
    return out


class PersonaSimulationTool(Tool):
    name = "PersonaSimulationTool"
    description = (
        "새로운 기업의 행보에 대해 가상 소비자의 페르소나를 반영하여 시장의 반응을 분석, 살펴보는 도구이다."
    )
    inputs = {
        "data_glob": {"type": "string", "description": "", "default": "./data/shopping/*.json", "nullable": True},
        "candidate_path": {"type": "string", "description": "", "default": "./inputs/candidates.json", "nullable": True},
        "evidence_path": {"type": "string", "description": "", "default": "./inputs/evidence.json", "nullable": True},
        "repeats": {"type": "integer", "description": "세트당 LLM 샘플 반복 횟수", "nullable": True},
        "temperature": {"type": "number", "description": "LLM temperature", "nullable": True},
        "seed": {"type": "integer", "description": "난수 시드", "nullable": True},
        "min_weight": {"type": "number", "description": "세그 최소 가중치(미만 제외, 없으면 top_k로 대체)", "nullable": True},
        "top_k": {"type": "integer", "description": "상위 세그먼트 수 제한", "nullable": True},
        "price_scale": {"type": "number", "description": "가격 스케일(원→스케일 단위)", "nullable": True},
        "preserve_set_id": {"type": "boolean", "description": "candidates의 set_id를 보존할지", "nullable": True},
        "match_mode": {"type": "string", "description": "증거-세그 매칭 모드: exact|partial", "nullable": True},
        "add_asc": {"type": "boolean", "description": "ASC 열 추가(N 비구매 성향 보정용)", "nullable": True},
        "asc_baseline": {"type": "string", "description": "ASC 기준 대안 키 (예:A). 지정 없으면 첫 대안", "nullable": True},
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_llm = None
        self.organizer = OpenAIServerModel(
            model_id="K-intelligence/Midm-2.0-Base-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="dummy-key",
            max_tokens=512,
            temperature=0
        )

    @staticmethod
    def _load_choice_sets(candidate_path: str, preserve_set_id: bool) -> List["ChoiceSet"]:
        with open(candidate_path, encoding="utf-8") as f:
            obj = json.load(f)
        out = []
        for cs in obj.get("choice_sets", []):
            alts = [ChoiceAlternative(a["key"], a["attributes"]) for a in cs["alternatives"]]
            if preserve_set_id:
                sid = cs.get("set_id") or str(uuid.uuid4())[:8]
            else:
                sid = str(uuid.uuid4())[:8]
            out.append(ChoiceSet(set_id=sid, alternatives=alts, include_no_purchase=bool(cs.get("include_no_purchase", True))))
        return out

    @staticmethod
    def _patch_partial_match(pb: "PersonaBuilderAgent"):
        def fuse_external_evidence_partial(segs, evidence_json, *, c=100, m=5, beta=1.0):
            if not segs: return segs
            w = np.array([max(1e-12, s.weight) for s in segs]); w=w/w.sum()
            eta = c*w
            s_vec = np.zeros(len(segs))
            def partial(seg: "PersonaSegment", flt: Dict[str,Any])->bool:
                for k, v in (flt or {}).items():
                    if v is None: 
                        continue
                    if k in seg.filters and seg.filters[k] is not None:
                        if seg.filters[k] != v:
                            return False
                return True
            for cl in evidence_json.get("claims", []):
                conf = float(cl.get("confidence", 0.0))
                rec = float(np.exp(-float(cl.get("recency_days", 365))/180.0))
                cons = float(np.log(1+int(cl.get("source_count",1)))/np.log(6))
                base = (conf + rec + cons) / 3.0
                signed = base if cl.get("direction", "+") != "-" else -base
                for i, seg in enumerate(segs):
                    if partial(seg, cl.get("segment", {})):
                        s_vec[i] += signed
            s_vec = 1/(1+np.exp(-beta*s_vec))
            eta_post = eta + m*s_vec
            w_post = eta_post/eta_post.sum()
            for i, seg in enumerate(segs):
                seg.weight = float(w_post[i])
                if s_vec[i] > 0:
                    seg.research_note = (seg.research_note + ", " if seg.research_note else "") + f"s={s_vec[i]:.2f}"
            return segs
        pb.fuse_external_evidence = fuse_external_evidence_partial.__get__(pb, pb.__class__)

    @staticmethod
    def _patch_add_asc(sim: "PersonaSimulatorAgent", asc_baseline: Optional[str] = None):
        if getattr(sim, "_asc_patched", False):
            return

        original_design_matrix = sim._design_matrix

        def _design_matrix_with_asc(self, cs: "ChoiceSet"):
            X, labels = original_design_matrix(cs)
            X = X.copy()

            for lab in labels:
                X[f"ASC_{lab}"] = 0.0

            base = asc_baseline if (asc_baseline and asc_baseline in labels) else labels[0]

            for idx, lab in enumerate(labels):
                if lab == base:
                    continue
                X.loc[idx, f"ASC_{lab}"] = 1.0

            return X, labels

        sim._design_matrix = MethodType(_design_matrix_with_asc, sim)
        sim._asc_patched = True

    def forward(
        self,
        data_glob: Optional[str] = "./data/shopping/*.json",     # ← 기본값으로 예시 경로
        candidate_path: Optional[str] = "./inputs/candidates.json",
        evidence_path: Optional[str] = "./inputs/evidence.json",
        repeats: Optional[int] = 100,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 42,
        min_weight: Optional[float] = 0.02,
        top_k: Optional[int] = 10,
        price_scale: Optional[float] = 100_000.0,
        preserve_set_id: Optional[bool] = True,
        match_mode: Optional[str] = "exact",
        add_asc: Optional[bool] = False,
        asc_baseline: Optional[str] = None,
    ) -> str:
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        pb = PersonaBuilderAgent(min_quality=0.0)
        if match_mode and match_mode.lower() == "partial":
            self._patch_partial_match(pb)

        segs = pb.build_segments(data_glob)
        if not segs:
            return json.dumps({"error": f"No segments built from {data_glob}. Check filters or data path."}, ensure_ascii=False)

        if evidence_path and pathlib.Path(evidence_path).exists():
            try:
                with open(evidence_path, encoding="utf-8") as f:
                    evidence = json.load(f)
                segs = pb.fuse_external_evidence(segs, evidence, c=100, m=5, beta=1.0)
            except Exception as e:
                pass

        try:
            if preserve_set_id:
                sets = self._load_choice_sets(candidate_path, preserve_set_id=True)
            else:
                sets = load_choice_sets_from_json(candidate_path)
        except Exception as e:
            return json.dumps({"error": f"Failed to load candidates: {repr(e)}"}, ensure_ascii=False)

        if not sets:
            return json.dumps({"error": "choice_sets is empty."}, ensure_ascii=False)

        attrs = {
            "통신사": ["KT", "SKT", "LGU+", "알뜰폰"],
            "데이터": ["10GB", "30GB", "무제한(속도제한)"],
            "통화문자": ["기본제공", "무제한"],
            "5G여부": ["4G", "5G"],
            "테더링": ["5GB", "20GB"],
            "로밍": ["미포함", "월 2GB"],
            "약정": ["무약정", "12개월"],
            "가족결합": ["미적용", "적용"],
            "OTT번들": ["없음", "웨이브", "티빙", "넷플릭스"]
        }
        price_grid = [19_000, 29_000, 39_000, 49_000, 59_000]

        sim = PersonaSimulatorAgent(
            attributes=attrs,
            price_grid=price_grid,
            num_alternatives=max(len(cs.alternatives) for cs in sets),
            include_no_purchase=any(cs.include_no_purchase for cs in sets),
            price_scale=float(price_scale or 100_000.0),
            organizer=self.organizer,
        )
        if add_asc:
            self._patch_add_asc(sim, asc_baseline=asc_baseline)

        result = run_multi_segment(
            sim, segs, sets,
            repeats=int(repeats or 100),
            temperature=float(temperature or 1.0),
            seed=int(seed or 42),
            min_weight=float(min_weight or 0.02),
            top_k=int(top_k or 10),
        )

        agg = result.get("aggregate", {})
        out = {
            "segments_used": agg.get("segments_used", []),
            "market_shares": agg.get("market_shares", {}),
            "market_wtp": agg.get("market_wtp", {}),
            "valid_rates": agg.get("valid_rates", {}),
        }
        return json.dumps(out, ensure_ascii=False, indent=2)


def main():
    modified_system_prompt = """당신은 도구 호출을 사용하여 모든 작업을 해결할 수 있는 전문 어시스턴트입니다. 해결해야 할 작업이 주어지면 최선을 다해 해결하게 됩니다. 답을 생각하고 액션하고 관찰하고 다시 생각하는 일련의 과정들을 거칩니다. 이를 위해 몇 가지 도구에 대한 접근 권한이 주어졌습니다. 작성하는 도구 호출은 하나의 액션입니다: 도구가 실행된 후, 도구 호출의 결과를 "관찰(observation)"로 받게 됩니다. 이 액션/관찰은 N번 반복될 수 있으며, 필요한 경우 여러 단계를 거쳐야 합니다. 이전 액션의 결과를 다음 액션의 입력으로 사용할 수 있습니다. 관찰은 항상 문자열이 될 것입니다: "image_1.jpg"와 같은 파일을 나타낼 수 있습니다. 그러면 이를 다음 액션의 입력으로 사용할 수 있습니다. 예를 들어 다음과 같이 할 수 있습니다:
        CopyObservation: "image_1.jpg"
        Action: {
            "name": "image_transformer",
            "arguments": {"image": "image_1.jpg"}
        }
        작업에 대한 최종 답변을 제공하려면 "name": "final_answer" 도구가 있는 액션 블록을 사용하세요. 이것이 작업을 완료하는 유일한 방법이며, 그렇지 않으면 루프에 갇히게 됩니다. 따라서 최종 출력은 다음과 같아야 합니다:
        CopyAction: {
            "name": "final_answer",
            "arguments": {"answer": "여기에 최종 답변 삽입"}
        }
        가상의 도구를 사용한 몇 가지 예시입니다:

        작업: "이 문서에서 가장 나이 많은 사람의 이미지를 생성하세요."
        CopyAction: {
            "name": "document_qa",
            "arguments": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}
        }
        Observation: "문서에서 가장 나이 많은 사람은 뉴펀들랜드에 사는 55세의 나무꾼 John Doe입니다."
        CopyAction: {
            "name": "image_generator",
            "arguments": {"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}
        }
        Observation: "image.png"
        CopyAction: {
            "name": "final_answer",
            "arguments": "image.png"
        }
        
        작업: "광저우와 상하이 중 어느 도시의 인구가 더 많습니까?"
        CopyAction: {
            "name": "search",
            "arguments": "Population Guangzhou"
        }
        Observation: ['광저우의 인구는 2021년 기준 1,500만 명입니다.']
        CopyAction: {
            "name": "search",
            "arguments": "Population Shanghai"
        }
        Observation: '2,600만 명 (2019)'
        CopyAction: {
            "name": "final_answer",
            "arguments": "Shanghai"
        }
        위의 예시들은 실제로 존재하지 않을 수 있는 가상의 도구를 사용했습니다. 당신은 다음 도구들에만 접근할 수 있습니다:
        {%- for tool in tools.values() %}

        {{ tool.name }}: {{ tool.description }}
        입력: {{tool.inputs}}
        출력 유형: {{tool.output_type}}
        {%- endfor %}

        {%- if managed_agents and managed_agents.values() | list %}
        팀 멤버들에게 요청을 할 수도 있습니다. 팀 멤버 호출은 도구 호출과 동일하게 작동합니다: 단순히 호출에서 제공할 수 있는 유일한 인수는 'request'로, 요청을 설명하는 긴 문자열입니다. 이 팀 멤버가 실제 사람이므로 요청 시 매우 자세히 설명해야 합니다. 호출할 수 있는 팀 멤버 목록은 다음과 같습니다:
        {%- for agent in managed_agents.values() %}

        {{ agent.name }}: {{ agent.description }}
        {%- endfor %}
        {%- else %}
        {%- endif %}

        작업을 해결하기 위해 항상 따라야 할 규칙은 다음과 같습니다:


        작업을 해결하기 위해 항상 따라야 할 규칙은 다음과 같습니다:

        반드시 도구 호출을 제공해야 합니다. 그렇지 않으면 실패합니다.
        도구에 올바른 인수를 사용하세요. 액션 인수로 변수 이름을 사용하지 말고 값을 사용하세요.
        필요한 경우에 도구를 호출하세요. 스스로 작업을 해결하려고 노력하세요. 
        많은 도구를 호출할 수록, 당신에게 더 많은 힌트를 안겨주고 이는 당신을 올바른 성공에 도달하게 합니다. 
        final_answer 도구 외에 반드시 당신에게 주어진 모든 도구를 한번 이상씩 호출하세요. 필요시 여러번 호출이 가능합니다.
        작업을 해결할 수 있을 때까지 계속 도구를 호출하세요.
        답을 내릴 수 있다면 final_answer 도구를 사용하여 답변을 반환하세요.
        final_answer도구 전 반드시 모든 도구를 호출했는지 확인하세요. 당신에게 주어진 도구 호출 기회는 7번으로 제한돼있습니다. 반드시 그 안에 주어진 도구 모두를 호출해야합니다
        이제 한국어를 사용하여 시작하세요!
        """
        
    query = "kt의 주력 사업에 대해 조사하여 신상품에 대한 아이디어를 추천받고 이에 대한 소비자 반응이 어떨지 시뮬레이션 돌려보고싶어."
    model = OpenAIServerModel(
        model_id="K-intelligence/Midm-2.0-Base-Instruct",
        api_base="http://localhost:8000/v1",
        api_key="dummy-key",
        max_tokens=512,
        temperature=0
    )
    
    search_tool = InternetSearchTool()
    persona_tool = PersonaSimulationTool()


    agent = ToolCallingAgent(
        tools=[search_tool, persona_tool],
        model=model, 
        max_steps=7,
        planning_interval = 1
    )
    agent.prompt_templates["system_prompt"] = modified_system_prompt

    agent_output = agent.run(query)
    print(agent_output)

        
if __name__ == "__main__":
    main()