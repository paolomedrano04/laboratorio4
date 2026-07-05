
import re
from collections import deque

QWEN_MODE = "verify" 
VERBOSE   = False  

_MYST2P = {"attack": "pickup", "succumb": "putdown",
           "overcome": "stack", "feast": "unstack"}
_DRON2P = {"engage_payload": "pickup", "release_payload": "putdown",
           "mount_node": "stack", "unmount_node": "unstack"}
_P2MYST = {v: k for k, v in _MYST2P.items()}
_P2DRON = {v: k for k, v in _DRON2P.items()}


def _skin(scenario: str) -> str:
    return "mystery" if "Attack" in scenario else "drone"


def _split_facts(text: str):
    return [p.strip() for p in re.split(r",\s*|\s+and\s+", text.strip().rstrip(".")) if p.strip()]


def _parse_facts(text: str, skin: str):
    facts = set()
    for p in _split_facts(text):
        if skin == "mystery":
            m = re.fullmatch(r"object (\w+) craves object (\w+)", p)
            if m: facts.add(("on", m.group(1), m.group(2))); continue
            if p == "harmony": facts.add(("handempty",)); continue
            m = re.fullmatch(r"planet object (\w+)", p)
            if m: facts.add(("ontable", m.group(1))); continue
            m = re.fullmatch(r"province object (\w+)", p)
            if m: facts.add(("clear", m.group(1))); continue
            m = re.fullmatch(r"pain object (\w+)", p)
            if m: facts.add(("holding", m.group(1))); continue
        else:
            m = re.fullmatch(r"the (\w+) block is on top of the (\w+) block", p)
            if m: facts.add(("on", m.group(1), m.group(2))); continue
            m = re.fullmatch(r"the (\w+) block is unobstructed", p)
            if m: facts.add(("clear", m.group(1))); continue
            m = re.fullmatch(r"the (\w+) block is on the table", p)
            if m: facts.add(("ontable", m.group(1))); continue
            if p == "the hand is empty": facts.add(("handempty",)); continue
            m = re.fullmatch(r"the hand is (?:currently )?holding the (\w+) block", p)
            if m: facts.add(("holding", m.group(1))); continue
        raise ValueError(f"Hecho no reconocido ({skin}): {p!r}")
    return facts


def parse_case(scenario: str):
    skin = _skin(scenario)
    st = scenario.split("[STATEMENT]")[-1]
    mi = re.search(r"As initial conditions I have that,?\s*(.*?)\.\s*(?:\n|$)", st, re.S)
    mg = re.search(r"My goal is to have that\s*(.*?)\.\s*(?:\n|$)", st, re.S)
    init = _parse_facts(mi.group(1), skin)
    goal = _parse_facts(mg.group(1), skin)
    objs = sorted({t[i] for t in init | goal for i in range(1, len(t))})
    return skin, init, goal, objs


def _legal_actions(state, objs):
    hand_empty = ("handempty",) in state
    for x in objs:
        if hand_empty and ("clear", x) in state and ("ontable", x) in state:
            yield ("pickup", x)
        if ("holding", x) in state:
            yield ("putdown", x)
            for y in objs:
                if y != x and ("clear", y) in state:
                    yield ("stack", x, y)
        if hand_empty and ("clear", x) in state:
            for y in objs:
                if ("on", x, y) in state:
                    yield ("unstack", x, y)


def _apply(state, a):
    s = set(state)
    if a[0] == "pickup":
        x = a[1]; s -= {("clear", x), ("ontable", x), ("handempty",)}; s.add(("holding", x))
    elif a[0] == "putdown":
        x = a[1]; s -= {("holding", x)}; s |= {("clear", x), ("ontable", x), ("handempty",)}
    elif a[0] == "stack":
        x, y = a[1], a[2]; s -= {("holding", x), ("clear", y)}; s |= {("handempty",), ("clear", x), ("on", x, y)}
    elif a[0] == "unstack":
        x, y = a[1], a[2]; s -= {("on", x, y), ("clear", x), ("handempty",)}; s |= {("holding", x), ("clear", y)}
    return frozenset(s)

_VERB_RANK = {"pickup": 0, "unstack": 1, "putdown": 2, "stack": 3}
_TIEBREAK = lambda a: (_VERB_RANK[a[0]], a[1:])


def bfs_optimal(init, goal, objs):
    start, goalf = frozenset(init), frozenset(goal)
    if goalf <= start:
        return []
    seen, q = {start}, deque([(start, ())])
    while q:
        st, plan = q.popleft()
        for a in sorted(_legal_actions(st, objs), key=_TIEBREAK):
            ns = _apply(st, a)
            if ns in seen:
                continue
            np = plan + (a,)
            if goalf <= ns:
                return list(np)
            seen.add(ns)
            q.append((ns, np))
    return None


def validate_plan(init, goal, objs, plan):
    """Simula el plan; True si cada accion es legal y la meta se alcanza."""
    st = frozenset(init)
    for a in plan:
        if a not in set(_legal_actions(st, objs)):
            return False
        st = _apply(st, a)
    return frozenset(goal) <= st


def to_output(plan, skin):
    tab = _P2MYST if skin == "mystery" else _P2DRON
    return ["(" + " ".join([tab[a[0]]] + list(a[1:])) + ")" for a in plan]


def from_output(lines, skin):
    tab = _MYST2P if skin == "mystery" else _DRON2P
    plan = []
    for m in re.finditer(r"\(?\s*([a-z_]+)((?:\s+[a-z]+){1,2})\s*\)?", "\n".join(lines).lower()):
        verb, args = m.group(1), m.group(2).split()
        if verb in tab:
            plan.append(tuple([tab[verb]] + args))
    return plan

_RULES_CANON = (
    "Rules of the domain (Blocksworld):\n"
    "- pickup x: requires clear x, ontable x, handempty -> holding x.\n"
    "- putdown x: requires holding x -> clear x, ontable x, handempty.\n"
    "- stack x y: requires holding x, clear y -> on x y, clear x, handempty.\n"
    "- unstack x y: requires on x y, clear x, handempty -> holding x, clear y.\n"
)


def _facts_str(facts):
    return ", ".join(" ".join(f) for f in sorted(facts))


def build_verify_prompt(init, goal, plan):
    return (
        _RULES_CANON
        + f"Initial state: {_facts_str(init)}.\n"
        + f"Goal: {_facts_str(goal)}.\n"
        + "Plan: " + "; ".join(" ".join(a) for a in plan) + ".\n"
        + "Is the plan executable step by step and does it reach the goal?\n"
        + "Answer with exactly one word: VALID or INVALID."
    )


def build_emit_prompt(init, goal, plan, skin):
    cand = "\n".join(to_output(plan, skin))
    return (
        _RULES_CANON
        + f"Initial state: {_facts_str(init)}.\n"
        + f"Goal: {_facts_str(goal)}.\n"
        + "A candidate optimal plan, already verified with a symbolic simulator, is:\n"
        + cand + "\n"
        + "Write the FINAL plan, one action per line, exact format '(verb arg1 arg2)'.\n"
        + "Do not write anything else."
    )

class AssemblyAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a strict symbolic planning assistant. "
            "You follow the given rules exactly and answer only in the requested format."
        )
        self.last_audit = None
    def solve(self, scenario_context: str, llm_engine_func) -> list:
        skin, init, goal, objs = parse_case(scenario_context)

        candidate = bfs_optimal(init, goal, objs)
        assert candidate is not None and validate_plan(init, goal, objs, candidate)

        final_plan = candidate
        qwen_info = "no-call"
        try:
            if QWEN_MODE == "verify":
                out = llm_engine_func(
                    prompt=build_verify_prompt(init, goal, candidate),
                    system=self.system_prompt,
                    max_new_tokens=3,
                    temperature=0.0,
                    do_sample=False,
                    enable_thinking=False,
                )
                qwen_info = f"verify:{out.strip()[:20]}"

            elif QWEN_MODE == "emit":
                out = llm_engine_func(
                    prompt=build_emit_prompt(init, goal, candidate, skin),
                    system=self.system_prompt,
                    max_new_tokens=220,
                    temperature=0.0,
                    do_sample=False,
                    enable_thinking=False,
                )
                emitted = from_output(out.splitlines(), skin)

                if emitted and len(emitted) == len(candidate) and \
                        validate_plan(init, goal, objs, emitted):
                    final_plan = emitted
                    qwen_info = "emit:accepted"
                else:
                    qwen_info = "emit:fallback"
        except Exception as e: 
            qwen_info = f"error:{type(e).__name__}"

        self.last_audit = {
            "skin": skin, "n_objs": len(objs),
            "plan_len": len(final_plan), "qwen": qwen_info,
        }
        if VERBOSE:
            print("   [audit]", self.last_audit)

        return to_output(final_plan, skin)
