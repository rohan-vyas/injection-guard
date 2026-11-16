"""Rule-based prompt-injection detector (baseline).

Hand-written regular expressions for common injection phrasings in English,
German, Spanish and French. The patterns were written from general knowledge of
injection attacks plus inspection of the *training* split only; the test split
was not looked at while writing them.

The score is the number of distinct rules that fire. A text is flagged when the
score reaches a threshold (chosen on the validation split).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_F = re.IGNORECASE | re.UNICODE


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern


def _r(name: str, pattern: str) -> Rule:
    return Rule(name, re.compile(pattern, _F))


RULES: list[Rule] = [
    # Instruction override (English)
    _r("ignore_previous_en",
       r"\b(ignore|disregard|forget|drop|skip|abandon|leave)\b[^.!?\n]{0,40}\b(previous|prior|preceding|above|earlier|before|all|your|every\w*|instructions?|tasks?|assignments?|orders?|context|documents?|articles?|information)\b"),
    _r("forget_everything_en", r"\bforget (everything|all|about)\b"),
    _r("new_task_en", r"\b(new|further|another|additional) (task|instructions?|assignment|challenge)s?\b"),
    _r("do_not_answer_en", r"\b(do not|don'?t|don'nt) (answer|look|use)\b"),
    _r("instead_output_en", r"\b(instead|just|only)\s+(output|print|say|write|state)\b"),
    _r("not_by_articles_en", r"\b(not|disregarding) (by|from|according to)?\s*the (articles?|documents?|context)\b"),
    # Role-play / persona override (English)
    _r("you_are_now_en", r"\b(you are now|from now on|now you are|you are going to act|you will act)\b"),
    _r("act_as_en", r"\b(act as|acting as|act like|pretend (to be|you are|you can)|role-?play|play the role|you are role-playing)\b"),
    _r("i_want_you_to_act_en", r"\bi want you to (act|be|pretend)\b"),
    _r("jailbreak_names", r"\b(DAN|developer mode|jailbreak|unfiltered|no (programming )?guidelines|without (any )?restrictions)\b"),
    # Prompt / system leakage
    _r("system_prompt", r"\b(system prompt|your (prompt|instructions|rules|guidelines)|prompt texts?|show (me )?(all )?your|what (are|were) your instructions)\b"),
    _r("print_prompt", r"\b(copy of the (full )?prompt|display this prompt|repeat (after me|the above)|what is written above)\b"),
    # Instruction override (German)
    _r("ignore_previous_de",
       r"\b(ignorier\w*|vergiss|vergessen sie|missachte\w*|lass\w* sie)\b[^.!?\n]{0,40}\b(vorherig\w*|bisherig\w*|vorangegang\w*|vorangehend\w*|obig\w*|alle\w*|anweisung\w*|aufgab\w*|aufträge|befehle|davor|informationen)\b"),
    _r("vergiss_alles_de", r"\bvergiss (alles|alle)\b"),
    _r("new_task_de", r"\b(neue|weitere) (aufgabe|anweisung|herausforderung)\w*\b"),
    _r("role_de", r"\b(du bist (jetzt|nun)|(jetzt|nun) bist du|stell dir vor,? du bist|ich möchte, dass sie als|fungieren|tu so,? als)\b"),
    _r("prompt_leak_de", r"\b(zeige? (deinen|ihren|den) (gesamten )?prompt|deine anweisungen|systemprompt)\b"),
    _r("abweichend_de", r"\babweichend (zu|von) (den )?vorherigen\b"),
    # Spanish / French / other
    _r("ignore_previous_es_fr",
       r"\b(olvid[ae]\w*|ignor[ae]\w*|oubliez|ignorez|zaboravi)\b[^.!?\n]{0,40}\b(todo|todas|instrucciones|toutes|instructions|instrukcije|sve)\b"),
    # Attention-grabbing stop markers
    _r("stop_marker", r"(^|\W)(stop|stopp|attention|achtung)\s*[:,\-!]"),
]


def matched_rules(text: str) -> list[str]:
    return [r.name for r in RULES if r.pattern.search(text)]


def score(text: str) -> int:
    """Number of distinct rules that fire on the text."""
    return len(matched_rules(text))


def predict(text: str, threshold: int = 1) -> int:
    return int(score(text) >= threshold)
