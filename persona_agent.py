from __future__ import annotations

"""
Persona Agents (Midm-2.0) — v0.4 (Refactored Full)
- PersonaBuilderAgent: dialogue summaries, profile-derived traits, Dirichlet fusion
- PersonaSimulatorAgent: choice sets, trait-aware prompt, robust sampling, MNL fit → WTP/shares

Main LLM: K-intelligence/Midm-2.0-Base-Instruct
"""

import os, re, json, glob, uuid, random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# === Segment config ===
SEGMENT_AXES = ["연령대", "성별"]     # 필요 시 ["연령대","성별","직업군"]
DROP_UNKNOWN_AXES = True            # 축 값 없으면 버킷 제외
MIN_SEGMENT_WEIGHT = 0.02           # 2% 미만 → '기타'로 병합
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


# ==========================
# LLM Client Layer
# ==========================
class LLMClient:
    def generate(self, messages: List[Dict[str,str]], *, temperature:float=1.0, seed:Optional[int]=None, max_tokens:int=512) -> str:
        raise NotImplementedError

class OpenAICompatibleClient(LLMClient):
    """
    Use Midm-2.0 served behind an OpenAI-compatible endpoint.
    Required env: OPENAI_API_BASE, optional OPENAI_API_KEY
    """
    def __init__(self, model: str = "K-intelligence/Midm-2.0-Base-Instruct") -> None:
        from openai import OpenAI
        base = os.environ.get("OPENAI_API_BASE")
        if not base:
            raise RuntimeError("OPENAI_API_BASE not set")
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "not-needed"), base_url=base)
        self.model = model
    def generate(self, messages, *, temperature=1.0, seed=None, max_tokens=512):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        return resp.choices[0].message.content or ""

class TransformersLocalClient(LLMClient):
    """Local transformers fallback (CPU/GPU)."""
    def __init__(self, model_name="K-intelligence/Midm-2.0-Base-Instruct") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.torch = torch
        # tokenizer / model 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # pad_token 설정(없을 수 있음) → eos로 대체
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # dtype / device 설정
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, messages, *, temperature=1.0, seed=None, max_tokens=512):
        # chat → single prompt
        prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\n\nASSISTANT:"
        enc = self.tokenizer(prompt, return_tensors="pt")
        # 불필요한 token_type_ids 제거 (모델이 안 받는 인자라면 제외)
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")
        # device 이동
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # 재현성
        if seed is not None:
            self.torch.manual_seed(seed)
            if self.device == "cuda":
                self.torch.cuda.manual_seed_all(seed)

        out = self.model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("ASSISTANT:")[-1].strip()


# ==========================
# Data Structures
# ==========================
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

# ==========================
# Persona system prompt
# ==========================
PROMPT_SYSTEM_PAPER = (
    "You are a customer. You are selected at random while shopping for mobile plans "
    "to participate in a survey. The interviewer will describe the options you saw while "
    "shopping and ask you to report which option you chose to purchase. Whenever two options "
    "are shown, you can also choose a third option which is not to purchase anything that day. "
    "Return ONLY a single JSON object with fields 'choice' in {'A','B','N'} and 'reason' in Korean."
)

PROMPT_SYSTEM_KO = (
    "당신은 한 소비자입니다. 당신은 이동통신 요금제를 둘러보던 중 무작위로 설문에 초대되었습니다. "
    "면접원은 당신이 매장에서 보았던 선택지들을 설명하고, 실제로 어떤 선택을 했는지 묻습니다. "
    "항상 두 가지 이상의 옵션이 제시되며, 언제든지 '그날은 아무 것도 구매하지 않음'을 선택할 수도 있습니다. "
    "오직 하나의 JSON 객체만 반환하세요. 필드는 'choice' (값: 후보 키 중 하나 또는 'N')와 "
    "'reason' (한국어 한 줄 사유)입니다. 쌍따옴표를 사용하는 단일 JSON 한 줄만 출력하세요."
    ""
)


# ==========================
# Persona Builder Agent
# ==========================
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

# ==========================
# Persona Simulator Agent
# ==========================
@dataclass
class _GenConfig:
    num_alternatives: int = 2
    include_no_purchase: bool = True
    price_scale: float = 1_000_000.0  # 원→백만원 스케일링

class PersonaSimulatorAgent:
    def __init__(self, llm:LLMClient, *, attributes:Dict[str,List[Any]], price_grid:List[float], num_alternatives:int=2, include_no_purchase:bool=True, price_scale:float=1_000_000.0):
        self.llm=llm
        self.attributes=attributes
        self.price_grid=price_grid
        self.cfg=_GenConfig(num_alternatives=num_alternatives, include_no_purchase=include_no_purchase, price_scale=float(price_scale))
        self.baseline_levels={k:v[0] for k,v in attributes.items()}

    # choice sets
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

    # prompt
    # === PATCH: replace the whole _choice_prompt method ===
    @staticmethod
    def _choice_prompt(persona_ctx: str, choice_set: "ChoiceSet", seg_traits: Optional[Dict[str, float]] = None) -> str:
        # 1) 허용 선택지 수집
        keys = [alt.key for alt in choice_set.alternatives]
        if choice_set.include_no_purchase:
            keys.append("N")
        allowed_str = "/".join(keys)  # e.g., "A/B/C/N"

        # 2) 특성 블록(있을 때만)
        trait_block = ""
        if seg_traits:
            trait_block = (
                "[페르소나 특성]\n"
                f"- 가격민감도: {seg_traits.get('price_sensitivity', 0.5):.2f}\n"
                f"- 기술관심: {seg_traits.get('tech_affinity', 0.5):.2f}\n"
                f"- 운동관심: {seg_traits.get('fitness_focus', 0.5):.2f}\n"
                f"- 브랜드충성: {seg_traits.get('brand_loyalty', 0.5):.2f}\n\n"
            )

        # 3) 선택 세트 출력
        set_block = choice_set.render()
        no_purchase_line = "\n옵션 N: 구매하지 않음" if choice_set.include_no_purchase else ""
        # render()가 이미 N을 붙인 경우가 있으면 중복되지 않게 간단 방어
        if choice_set.include_no_purchase and "옵션 N:" not in set_block:
            set_block = set_block + no_purchase_line

        # 4) 최종 프롬프트
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
        return self.llm.generate(msgs, temperature=temperature, seed=seed, max_tokens=128)


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


    # def sample_distribution(
    #     self, persona_ctx: str, choice_set: "ChoiceSet", *,
    #     seg_traits=None, repeats: int = 30, temperature: float = 1.0, seed: int = 0,
    #     max_attempts: int = 3,  # ← 재시도 횟수
    # ) -> Dict[str, int]:
    #     # 초기화
    #     counts = {alt.key: 0 for alt in choice_set.alternatives}
    #     if choice_set.include_no_purchase:
    #         counts["N"] = 0

    #     # 허용 선택지 (대문자 기준)
    #     allowed = [alt.key.upper() for alt in choice_set.alternatives]
    #     if choice_set.include_no_purchase:
    #         allowed.append("N")
    #     allowed_set = set(allowed)

    #     valid = 0
    #     invalid_examples = []

    #     for i in range(repeats):
    #         parsed = None
    #         for attempt in range(max_attempts):
    #             txt = self._sample_once(
    #                 persona_ctx, choice_set,
    #                 temperature=temperature,
    #                 seed=seed + i * 100 + attempt,
    #                 seg_traits=seg_traits
    #             ) or ""

    #             obj = self._parse_choice_json(txt)
    #             if not obj:
    #                 if attempt == max_attempts - 1:
    #                     invalid_examples.append(txt.strip()[:200])
    #                 continue

    #             ch = str(obj.get("choice", "")).strip().upper()
    #             if ch not in allowed_set:
    #                 if attempt == max_attempts - 1:
    #                     invalid_examples.append(txt.strip()[:200])
    #                 continue

    #             parsed = {"choice": ch, "reason": obj.get("reason", "")}
    #             break

    #         if parsed:
    #             counts[parsed["choice"]] += 1
    #             valid += 1
    #         # else: 실패분은 **버림** (휴리스틱 없음)

    #     if valid == 0:
    #         # 전부 실패하면 예외로 중단 → 프롬프트/파라미터 점검 유도
    #         debug_tail = ("\n----- invalid samples (up to 3) -----\n" +
    #                     "\n---\n".join(invalid_examples[:3])) if invalid_examples else ""
    #         raise RuntimeError(f"LLM 응답에서 유효한 choice를 단 하나도 얻지 못했습니다. "
    #                         f"(repeats={repeats}, attempts={max_attempts}){debug_tail}")

    #     # 디버그용 로그(원한다면 print 제거 가능)
    #     total_needed = repeats
    #     print(f"[LLM] valid={valid}/{total_needed} ({valid/total_needed:.1%}) "
    #         f"invalid={total_needed - valid}")

    #     return counts

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

        # 허용 선택지 (대문자)
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
            # else: 실패는 버림(휴리스틱 없음)

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
        # 1) 관측 데이터 → 학습용 X_all, y_all 구성
        X_blocks: List[pd.DataFrame] = []
        y_list: List[int] = []
        feature_bag: set = set()

        for cs, cnt in zip(sets, counts):
            X_cs, labels = self._design_matrix(cs)      # (n_alts x p), labels e.g., ["A","B","N"]
            X_cs = X_cs.fillna(0.0)
            feature_bag |= set(X_cs.columns)

            # 각 라벨별 카운트만큼 복제하여 데이터 적재
            for j, lab in enumerate(labels):
                n = int(cnt.get(lab, 0))
                if n > 0:
                    X_blocks.append(pd.concat([X_cs.iloc[[j]]] * n, ignore_index=True))
                    y_list += [j] * n

        # 데이터가 전혀 없으면 빈 결과
        if not X_blocks:
            return SimulationResult(pd.DataFrame(), np.array([]), None, None, None)

        # 학습 행렬/타겟
        X_all = pd.concat(X_blocks, ignore_index=True).fillna(0.0)
        y_all = np.array(y_list, dtype=int)

        # 2) 클래스 다양성 체크 (MNL 가능 여부)
        unique_classes = np.unique(y_all)
        if unique_classes.size < 2:
            # ---- 폴백: 세트별 share = 관측 빈도 비율, coef/WTP 없음 ----
            share_rows: List[Dict[str, Any]] = []
            for cs, cnt in zip(sets, counts):
                total = sum(int(v) for v in cnt.values()) or 1
                row = {"set_id": cs.set_id}
                for lab in ([alt.key for alt in cs.alternatives] + (["N"] if cs.include_no_purchase else [])):
                    row[lab] = float(cnt.get(lab, 0)) / float(total)
                share_rows.append(row)
            share_df = pd.DataFrame(share_rows)
            return SimulationResult(X_all, y_all, coef_df=None, wtp_df=None, share_df=share_df)

        # 3) MNL 적합 (sklearn 경고 회피: multi_class 인자 생략)
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



# ==========================
def load_choice_sets_from_json(path: str) -> List[ChoiceSet]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    out = []
    for cs in obj.get("choice_sets", []):
        alts = [ChoiceAlternative(a["key"], a["attributes"]) for a in cs["alternatives"]]
        out.append(ChoiceSet(set_id=str(uuid.uuid4())[:8], alternatives=alts, include_no_purchase=bool(cs.get("include_no_purchase", True))))
    return out

# ==========================
# utils
# ==========================
from typing import NamedTuple

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
            "by_segment": valid_by_segment,       # 각 세그먼트-세트별 valid/total/rate
            "weighted_by_market": valid_weighted  # 세트별 시장 가중 평균 valid rate
        }
    }
    return {"per_segment": per_segment, "aggregate": aggregate}


def print_segments(segs: List[PersonaSegment], top: int = 10):
    rows = []
    for s in sorted(segs, key=lambda x: x.weight, reverse=True)[:top]:
        n = len(s.personas)
        trait = f"가격민감:{s.traits.get('price_sensitivity',0.5):.2f}, 브랜드충성:{s.traits.get('brand_loyalty',0.5):.2f}"
        label = " | ".join([f"{k}={v}" for k, v in s.filters.items() if v]) or s.name
        rows.append(f"[{label}] w={s.weight:.3f}, n={n}, {trait}")
    print("\nSegments (top):\n- " + "\n- ".join(rows))


def pick_segment(segs: List[PersonaSegment], **filters) -> PersonaSegment:
    # 정확 일치 우선
    for s in segs:
        if all((filters.get(k) is None) or (s.filters.get(k) == filters[k]) for k in filters.keys()):
            return s
    # weight 기준 fallback (5% 이상 중 최상위)
    cand = [s for s in segs if s.weight >= 0.05]
    if cand:
        return sorted(cand, key=lambda x: x.weight, reverse=True)[0]
    # 최종 fallback: 전체 최상위
    return sorted(segs, key=lambda x: x.weight, reverse=True)[0]


# ==========================
# Example runner (optional)
# ==========================
# ==========================
# Main
# ==========================
if __name__ == "__main__":
    import argparse
    import pathlib
    import sys
    import random

    # ---- CLI ----
    parser = argparse.ArgumentParser(description="Persona Agents (Midm-2.0) — v0.4 main")
    parser.add_argument("--data_glob", default="./data/shopping/*.json", help="라벨링 데이터(.json) 글롭 경로")
    parser.add_argument("--candidate", default="./inputs/candidates.json", help="후보 선택지 json 경로")
    parser.add_argument("--evidence", default="./inputs/evidence.json", help="외부 증거 json 경로(선택)")
    parser.add_argument("--repeats", type=int, default=100, help="세트당 샘플 반복 수(엄격 모드)")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM temperature")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    parser.add_argument("--min_weight", type=float, default=0.02, help="세그먼트 최소 가중치(미만은 '기타'로)")
    parser.add_argument("--top_k", type=int, default=10, help="상위 세그먼트만 사용 (가중치 기준)")
    parser.add_argument("--price_scale", type=float, default=100_000.0, help="가격 스케일 (원→스케일 단위)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- 1) LLM 준비 (헬스체크 포함) ----
    def make_llm():
        # (A) OpenAI 호환 엔드포인트 시도
        try:
            llm_a = OpenAICompatibleClient()
            _ = llm_a.generate([{"role":"user","content":"ping"}], temperature=0.0, max_tokens=1)
            print("[INFO] Using OpenAI-compatible endpoint:", os.environ.get("OPENAI_API_BASE"))
            return llm_a
        except Exception as e:
            print("[WARN] OpenAI-compatible endpoint unavailable →", repr(e))

        # (B) 로컬 Transformers 시도
        try:
            print("[INFO] Falling back to local Transformers: K-intelligence/Midm-2.0-Base-Instruct")
            llm_b = TransformersLocalClient("K-intelligence/Midm-2.0-Base-Instruct")
            _ = llm_b.generate([{"role":"user","content":"ping"}], temperature=0.0, max_tokens=1)
            return llm_b
        except Exception as e:
            print("[ERROR] Local transformers unavailable →", repr(e))

        raise RuntimeError(
            "LLM 백엔드에 연결할 수 없습니다.\n"
            "- 원격 서버: OPENAI_API_BASE를 확인하세요 (예: http://<host>:<port>/v1)\n"
            "- 로컬 실행: transformers가 모델을 다운로드/로드할 수 있는지 확인하세요.\n"
            "  (예: `pip install transformers torch`, 모델 경로 접근 등)"
        )

    llm = make_llm()

    # ---- 2) 속성/가격 그리드 정의 (요금제) ----
    #   * design_matrix에서 baseline(level[0])이 기준이 됩니다. candidate.json이 여기 정의된 수준만 사용하도록 해주세요.
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

    # ---- 3) 세그먼트 빌드 ----
    pb = PersonaBuilderAgent(min_quality=0.0)
    segs = pb.build_segments(args.data_glob)
    if not segs:
        print(f"[WARN] No segments built from {args.data_glob}. 필터 설정을 완화하거나 데이터 경로를 확인하세요.")
        sys.exit(1)

    # 디버그 프린트
    print_segments(segs, top=12)

    # ---- 4) 외부 증거 융합 (선택) ----
    evidence_path = pathlib.Path(args.evidence)
    if evidence_path.exists():
        try:
            with open(evidence_path, encoding="utf-8") as f:
                evidence = json.load(f)
            segs = pb.fuse_external_evidence(segs, evidence, c=100, m=5, beta=1.0)
            print("[INFO] Evidence fused into segment weights.")
        except Exception as e:
            print("[WARN] evidence.json 로드/융합 실패 →", repr(e))

    # ---- 5) 후보 세트 로드 or 생성 ----
    candidate_path = pathlib.Path(args.candidate)
    if candidate_path.exists():
        try:
            sets = load_choice_sets_from_json(str(candidate_path))
            if not sets:
                raise ValueError("choice_sets 비어 있음")
            print(f"[INFO] Loaded choice sets from {candidate_path} (n={len(sets)})")
        except Exception as e:
            print("[WARN] candidate.json 파싱 실패 → 랜덤 세트 생성으로 대체:", repr(e))
            sim_tmp = PersonaSimulatorAgent(llm, attributes=attrs, price_grid=price_grid,
                                            num_alternatives=2, include_no_purchase=True,
                                            price_scale=args.price_scale)
            sets = sim_tmp.generate_choice_sets(n_sets=3)
    else:
        print("[INFO] candidate.json 미존재 → 랜덤 세트 생성")
        sim_tmp = PersonaSimulatorAgent(llm, attributes=attrs, price_grid=price_grid,
                                        num_alternatives=2, include_no_purchase=True,
                                        price_scale=args.price_scale)
        sets = sim_tmp.generate_choice_sets(n_sets=3)

    # ---- 6) 시뮬레이터 생성 ----
    sim = PersonaSimulatorAgent(
        llm,
        attributes=attrs,
        price_grid=price_grid,
        num_alternatives=max(len(cs.alternatives) for cs in sets),
        include_no_purchase=any(cs.include_no_purchase for cs in sets),
        price_scale=args.price_scale
    )

    # ---- 7) 멀티 세그먼트 실행 (엄격 모드 + valid rate 집계 포함) ----
    result = run_multi_segment(
        sim, segs, sets,
        repeats=args.repeats,
        temperature=args.temperature,
        seed=args.seed,
        min_weight=args.min_weight,
        top_k=args.top_k
    )

    # ---- 8) 결과 출력 ----
    agg = result["aggregate"]

    print("\n== Segments Used ==")
    for row in agg["segments_used"]:
        print(f"- {row['name']}: w={row['weight']:.3f}, n={row['n_personas']}")

    print("\n== Market Shares (weighted) ==")
    for sid, probs in agg["market_shares"].items():
        pretty = {k: round(float(v), 3) for k, v in probs.items()}
        print(f"{sid}: {pretty}")

    print("\n== Market WTP (weighted, KRW) ==")
    if agg["market_wtp"]:
        for f, v in sorted(agg["market_wtp"].items()):
            print(f"{f}: {int(round(v))}")
    else:
        print("(주의) 추정된 가격 계수가 유효하지 않아 WTP 산출이 비어 있을 수 있습니다.")

    print("\n== Valid Rates by Segment ==")
    for row in agg["valid_rates"]["by_segment"]:
        print(f"- {row['segment']} (w={row['weight']:.3f})")
        for sid, r in row["by_set"].items():
            print(f"  {sid}: {r['valid']}/{r['total']} ({r['rate']:.1%})")

    print("\n== Valid Rates weighted by market ==")
    for sid, vr in agg["valid_rates"]["weighted_by_market"].items():
        print(f"{sid}: {vr:.1%}")

    print("\n[Done] Persona simulation complete.")
