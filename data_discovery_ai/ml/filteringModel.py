# The data delivery mode filter model to classify the metadata records based on their titles, abstracts, and lineages.
# decision logic documented here: https://utas.atlassian.net/wiki/spaces/IMOS/pages/1353023492/Delivery+Mode+Classfication
import numpy as np
import structlog
import re
from typing import Any, Dict, List, Optional, Pattern
from dataclasses import dataclass, field
import tensorflow as tf
from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.enum.delivery_mode_enum import UpdateFrequency

logger = structlog.get_logger(__name__)

@dataclass
class DeliveryModePatterns:
    # explicit real-time mode identifiers: {near real time, real time, nrt, delayed}
    real_time_mode: Pattern = field(default_factory=lambda: re.compile(
        r"\b(near[-\s]?real[-\s]?time|real[-\s]?time|nrt)\b", re.I
    ))

    # explicit delayed mode identifiers: {delayed, delaying, delay}
    delayed_mode: Pattern = field(default_factory=lambda: re.compile(
        r"\b(delayed(?:[-\s]?mode)?)\b", re.I
    ))

    # post-processing identifiers, e.g., quality control, processed, etc., used for identifying delayed mode
    processed: Pattern = field(default_factory=lambda: re.compile(
        r"\b(quality[-\s]?controlled|quality control|qc\b|qc[-\s]?flags?|flagged|validated|processed|reprocessed|post[-\s]?processed)\b", re.I
    ))

    # delivery verb identifiers
    delivery_verb: Pattern = field(default_factory=lambda: re.compile( r"\b(via|available|provided|delivered|uploaded|broadcast|published|accessible)\b", re.I ))

    # unprocessed identifiers, e.g., raw data, used for identifying real-time mode
    unprocessed: Pattern = field(default_factory=lambda: re.compile(
        r"\b(unprocessed|raw|no quality control|no qc|no quality control flags|without qc flags|not quality controlled)\b", re.I
    ))

    # target identifier, i.e., specifically indicates this record, so that to make sure the text explicitly indicates current record.
    target: Pattern = field(default_factory=lambda: re.compile(
        r"\b(this (dataset|record|data|product|collection)|"
        r"the (data|observations|measurements)|"
        r"(data|observations|measurements) (is|are))\b",
        re.I
    ))

    # Contrast markers for clause selection. Edge case: "the rest of the collection are in delayed mode, while this one is delivered in real-time"
    contrast: Pattern = field(default_factory=lambda: re.compile(
        r"\b(but|however|whereas|although|instead|unlike|except|rather than|while)\b", re.I
    ))

@dataclass(frozen=True)
class DeliveryModeHypothesis:
    subject: str = "This record"

    @property
    def delayed(self) -> str:
        return f"{self.subject} is delivered in delayed mode."

    @property
    def real_time(self) -> str:
        return f"{self.subject} is delivered in real-time mode."

@dataclass
class Evidence:
    rt: List[str] = field(default_factory=list)
    delayed: List[str] = field(default_factory=list)
    rt_unprocessed: List[str] = field(default_factory=list)

def extract_evidence(title: str, description: str, statement: str,
                     max_rt: int = 3, max_del: int = 3, max_unp: int = 2) -> Evidence:
    """
    Extract evidence sentences from title, description, and statement that indicate whether the record is real-time or delayed mode.
    Filters out noise and keeps only relevant clauses as evidence.
    Inputs:
        - title: str, the title of the record
        - description: str, the description of the record
        - statement: str, the statement of the record
        - max_rt: int, the maximum number of evidence sentences for supporting real-time to return
        - max_del: int, the maximum number of evidence sentences for supporting delayed to return
        - max_unp: int, the maximum number of evidence sentences for supporting unprocessed to return
    Outputs:
        Evidence: an evidence object to store evidence sentences in rt, delayed, and rt_unprocessed lists.
    """
    ev = Evidence()
    p = DeliveryModePatterns()
    def process_field(text: str):
        for sent in clean_text_to_sentences(text):
            clauses = split_clauses(sent)

            # contrast: prefer tail clause that contains target
            if p.contrast.search(sent) and len(clauses) > 1:
                pick = None
                for c in reversed(clauses):
                    if p.target.search(c):
                        pick = c
                        break
                clauses = [pick] if pick else clauses[-2:]

            for c in clauses:
                has_target = bool(p.target.search(c))
                has_delivery_verb = bool(p.delivery_verb.search(c))
                has_rt = bool(p.real_time_mode.search(c))
                has_del = bool(p.delayed_mode.search(c))
                has_proc = bool(p.processed.search(c))
                has_unp = bool(p.unprocessed.search(c))

                # REAL-TIME evidence:
                # (a) explicit real-time mode assertion
                if has_rt and has_delivery_verb and len(ev.rt) < max_rt:
                    ev.rt.append(c)
                # (b) implicit real-time mode assertion: unprocessed implies real-time
                if has_target and has_unp and len(ev.rt) < max_unp:
                    ev.rt.append(c)

                # DELAYED evidence:
                # (a) explicit delayed mode assertion
                if has_del and has_delivery_verb and len(ev.delayed) < max_del:
                    ev.delayed.append(c)
                # (b) processed implies delayed (as evidence)
                elif has_proc and (not has_unp) and len(ev.delayed) < max_del:
                    ev.delayed.append(c)

                # CONFLICT: rt + delivery + unprocessed
                if has_rt and has_unp and len(ev.rt_unprocessed) < max_unp:
                    ev.rt_unprocessed.append(c)

                if len(ev.rt) >= max_rt and len(ev.delayed) >= max_del and len(ev.rt_unprocessed) >= max_unp:
                    return

    process_field(title)
    process_field(description)
    process_field(statement)

    ev.rt = remove_duplicated_sentences(ev.rt)
    ev.delayed = remove_duplicated_sentences(ev.delayed)
    ev.rt_unprocessed = remove_duplicated_sentences(ev.rt_unprocessed)
    return ev

def clean_text_to_sentences(text: str) -> List[str]:
    sentence_split = re.compile(r"(?<=[\.\!\?\;])\s+|\n+")
    if text is None:
        return []
    # remove spaces
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return [x.strip() for x in sentence_split.split(text) if x and x.strip()]

def split_clauses(sent: str) -> List[str]:
    clause_split = re.compile(r"\b(?:but|however|whereas|although|instead|unlike|except|rather than|while)\b|[,;]", re.I)
    parts = [x.strip() for x in re.split(clause_split, sent) if x and x.strip()]
    return parts if parts else [sent.strip()]

def remove_duplicated_sentences(sentences: List[str]) -> List[str]:
    out, seen = [], set()
    for sentence in sentences:
        lower_case_sent = sentence.lower()
        if lower_case_sent not in seen:
            seen.add(lower_case_sent)
            out.append(sentence)
    return out


class DeliveryModeInferencer:
    def __init__(self, tokenizer, model):
        self.config = ConfigUtil.get_config().get_delivery_trainer_config()
        self.tokenizer = tokenizer
        self.model = model

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

        self.ent_id = label2id.get("entailment", None)
        self.con_id = label2id.get("contradiction", None)
        self.neu_id = label2id.get("neutral", None)
        if self.ent_id is None or self.con_id is None:
            raise ValueError(f"Unexpected labels: {self.model.config.id2label}")

    def calculate_probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="tf",
            truncation="only_first",
            max_length=self.config.max_length,
        )
        logits = self.model(inputs).logits
        probs = tf.nn.softmax(logits, axis=-1).numpy().squeeze(0)
        out = {"entailment": float(probs[self.ent_id]), "contradiction": float(probs[self.con_id]),
               "neutral": float(probs[self.neu_id]) if self.neu_id is not None else float(np.nan)}
        return out

    def decide_with_nli(
            self,
            ev: Evidence,
            max_sents: int = 5,
    ) -> Dict[str, Any]:
        hypothesis = DeliveryModeHypothesis()
        def max_entailment(sentences: List[str], hypo: str):
            scores = []
            for s in sentences[:max_sents]:
                s = (s or "").strip()
                if not s:
                    continue
                probs = self.calculate_probs(s, hypo)
                scores.append(probs["entailment"])
            return max(scores) if scores else 0.0, scores

        ent_rt, rt_scores = max_entailment(ev.rt, hypothesis.real_time)
        ent_dl, dl_scores = max_entailment(ev.delayed, hypothesis.delayed)

        if not ev.rt and not ev.delayed:
            return {
                "mode": UpdateFrequency.UNKNOWN.value,
                "reason": "no_evidence",
                "evidence": ev,
                "nli": None,
            }

        # Multi-product detection
        multi_product = (len(ev.rt_unprocessed) > 0) and (len(ev.delayed) > 0)

        # Case 1: both high - max(entailment_delayed) >= threshold entailment_high and max(entailment_real_time) >= threshold entailment_high
        # (a) if indicates unprocessed and delayed => REAL-TIME
        # (b) else => CONFLICT
        if ent_dl >= self.config.conflict_high and ent_rt >= self.config.conflict_high:
            return {
                "mode": UpdateFrequency.REAL_TIME.value if multi_product else UpdateFrequency.BOTH.value,
                "reason": "both_entail_high_multi_product" if multi_product else "both_entail_high_conflict",
                "secondary": "REAL_TIME_UNPROCESSED_AVAILABLE" if multi_product else None,
                "evidence": ev,
                "nli": {
                    "ent_rt": ent_rt,
                    "ent_dl": ent_dl,
                    "rt_scores": rt_scores,
                    "dl_scores": dl_scores,
                },
            }

        # Case 2: delayed, max(entailment_delayed) >= threshold entailment_high and max(entailment_real_time) <= threshold entailment_low
        if ent_dl >= self.config.entailment_high and ent_rt <= self.config.entailment_low:
            return {
                "mode": UpdateFrequency.DELAYED.value,
                "reason": "nli_entails_delayed",
                "evidence": ev,
                "nli": {
                    "ent_rt": ent_rt,
                    "ent_dl": ent_dl,
                    "rt_scores": rt_scores,
                    "dl_scores": dl_scores,
                },
            }

        # Case 3: real-time, max(entailment_real_time) > threshold entailment_high and max(entailment_delayed) < threshold entailment_low
        if ent_rt >= self.config.entailment_high and ent_dl <= self.config.entailment_low:
            return {
                "mode": UpdateFrequency.REAL_TIME.value,
                "reason": "nli_entails_real_time",
                "evidence": ev,
                "nli": {
                    "ent_rt": ent_rt,
                    "ent_dl": ent_dl,
                    "rt_scores": rt_scores,
                    "dl_scores": dl_scores,
                },
            }

        # Case 4: unknow if uncertain
        return {
            "mode": UpdateFrequency.UNKNOWN.value,
            "reason": "nli_abstain_below_threshold",
            "evidence": ev,
            "nli": {
                "ent_rt": ent_rt,
                "ent_dl": ent_dl,
                "rt_scores": rt_scores,
                "dl_scores": dl_scores,
            },
        }


def map_title_update_frequency(title: str) -> Optional[str]:
    real_time_variants = ["real time", "real-time", "realtime"]
    delayed_variants = ["delayed", "delay", "delaying"]
    # if real-time variants in title, return real-time mode
    if re.search("|".join(real_time_variants), title, re.IGNORECASE):
        return UpdateFrequency.REAL_TIME.value
    # if delayed variants in title, return delayed mode
    if re.search("|".join(delayed_variants), title, re.IGNORECASE):
        return UpdateFrequency.DELAYED.value
    # default as null
    return None


def mapping_update_frequency(status: str, temporal: list, title: str) -> str:
    """
    Input:
        - status: str - the status of the record, such as "completed", "onGoing", can have free text in one or few words.
        - temporal: List<Map>, for example:
        "temporal": [
            {
                "start": "2023-01-22T13:00:00Z",
                "end": "2023-01-23T12:59:59Z"
            },
            {}
        ]
    Output:
        str: the mapped update frequency in terms of "completed" or "other"
    Given status and temporal range of the record, decide update_frequency based on a set of predefined rules
    (see: https://github.com/aodn/backlog/issues/7978#issuecomment-3821164737). Specifically:
    1. these statuses could also be regarded as 'completed'
        historicalArchive
        obsolete
        deprecated
        complete
        Complete
    2. onGoing | historicalArchive - records have 2 x status identifed, 'ongoing' is priority
    3. Under development = check has date range. Yes = completed
        Planned = check has date range. Yes = completed
        Tentative = check has date range. Yes = completed
        No status = check has date range. Yes = completed (likely they are completed)
    Set update_frequency to 'completed' if meet rules or 'other' if not meet
    4. for records that with no "completed" status, if "real-time" or "delayed" in title, map as "real-time" or "delayed"
    """
    if status is None:
        status = ""
    normalised_status = status.replace(" ", "").lower()

    mapped_mode = map_title_update_frequency(title)
    # if status is completed, return 'completed' delivery mode
    if status.lower() == UpdateFrequency.COMPLETED.value:
        return UpdateFrequency.COMPLETED.value
    # rule 2: check ongoing priority
    if "ongoing" in normalised_status:
        # rule 4: check title for real-time/delayed variants
        if mapped_mode:
            return mapped_mode
        return UpdateFrequency.OTHER.value
    # rule 1: check completed status with its variants, if matched, return 'completed' delivery mode
    completed_status = ["historicalarchive", "obsolete", "deprecated", "complete"]
    if normalised_status in completed_status:
        return UpdateFrequency.COMPLETED.value
    # rule 3: check free text status with temporal range
    free_text_status = ["underdevelopment", "planned", "tentative", ""]
    if normalised_status in free_text_status and temporal is not None:
        for temporal_entry in temporal:
            if temporal_entry.get("start") and temporal_entry.get("end"):
                return UpdateFrequency.COMPLETED.value
        if mapped_mode:
            return mapped_mode

    # default to 'other'
    return UpdateFrequency.OTHER.value