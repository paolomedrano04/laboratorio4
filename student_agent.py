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
_FACT_VERBS = {"on": 2, "clear": 1, "ontable": 1, "handempty": 0, "holding": 1}


def _skin(s):
    return "mystery" if "Attack" in s else "drone"


def _statement_sentences(scenario):
    """Frases de estado inicial y meta del ULTIMO [STATEMENT] (texto crudo
    que se le pasa al LLM en la Etapa A)."""
    st = scenario.split("[STATEMENT]")[-1]
    mi = re.search(r"As initial conditions I have that,?\s*(.*?)\.\s*(?:\n|$)", st, re.S)
    mg = re.search(r"My goal is to have that\s*(.*?)\.\s*(?:\n|$)", st, re.S)
    return mi.group(1).strip(), mg.group(1).strip()

def _split_facts(text):
    return [p.strip() for p in re.split(r",\s*|\s+and\s+", text.strip().rstrip(".")) if p.strip()]


def _parse_facts_grammar(text, skin):
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

_EXTRACT_SYSTEM = ("You are a semantic parser for planning problems. "
                   "You translate natural language into canonical facts.")

_EXTRACT_DEMO_MYSTERY = """Vocabulary mapping: 'x craves y'->on x y, 'province x'->clear x, 'planet x'->ontable x, 'harmony'->handempty, 'pain x'->holding x.

Example:
Initial text: object b craves object a, harmony, planet object a, planet object c, province object b and province object c
Goal text: object a craves object c
Reasoning: b craves a is on b a; harmony is handempty; planets a,c are ontable; provinces b,c are clear.
INIT: on b a; handempty; ontable a; ontable c; clear b; clear c
GOAL: on a c

Now do the same for:
Initial text: {init}
Goal text: {goal}
Answer with one short 'Reasoning:' line, then exactly one 'INIT:' line and one 'GOAL:' line."""

_EXTRACT_DEMO_DRONE = """Vocabulary mapping: 'x is on top of y'->on x y, 'x is unobstructed'->clear x, 'x is on the table'->ontable x, 'the hand is empty'->handempty, 'holding x'->holding x. Use the block color as the object name.

Example:
Initial text: the red block is unobstructed, the hand is empty, the red block is on top of the blue block, the blue block is on the table and the orange block is on the table
Goal text: the blue block is on top of the orange block
Reasoning: red on blue; blue and orange on table; red clear; hand empty. Wait, orange has nothing on top so it is also clear? Only listed facts count: clear red only.
INIT: clear red; handempty; on red blue; ontable blue; ontable orange
GOAL: on blue orange

Now do the same for:
Initial text: {init}
Goal text: {goal}
Answer with one short 'Reasoning:' line, then exactly one 'INIT:' line and one 'GOAL:' line."""


def build_extract_prompt(init_txt, goal_txt, skin):
    tpl = _EXTRACT_DEMO_MYSTERY if skin == "mystery" else _EXTRACT_DEMO_DRONE
    return tpl.format(init=init_txt, goal=goal_txt)


def parse_llm_facts(out):
    """Parsea las lineas INIT:/GOAL: emitidas por Qwen a conjuntos de hechos.
    Devuelve (init, goal) o None si la salida es malformada."""
    def line(tag):
        m = re.search(rf"{tag}\s*:\s*(.+)", out, re.I)
        return m.group(1).strip() if m else None
    li, lg = line("INIT"), line("GOAL")
    if not li or not lg:
        return None
    def facts(l):
        fs = set()
        for chunk in re.split(r"[;,]\s*", l.strip().rstrip(".")):
            toks = chunk.strip().lower().split()
            if not toks:
                continue
            if toks[0] not in _FACT_VERBS or len(toks) - 1 != _FACT_VERBS[toks[0]]:
                return None
            fs.add(tuple(toks))
        return fs
    fi, fg = facts(li), facts(lg)
    return (fi, fg) if fi and fg else None


def _sane(init, goal):
    """Chequeo estructural minimo antes de planificar sobre la extraccion."""
    hold = [f for f in init if f[0] == "holding"]
    if (("handempty",) in init) == bool(hold):
        return False
    objs = {t for f in init | goal for t in f[1:]}
    return bool(objs) and all(f[0] == "on" for f in goal)


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
            if goalf <= ns:
                return list(plan) + [a]
            seen.add(ns)
            q.append((ns, plan + (a,)))
    return None


def validate_plan(init, goal, objs, plan):
    st = frozenset(init)
    for a in plan:
        if a not in set(_legal_actions(st, objs)):
            return False
        st = _apply(st, a)
    return frozenset(goal) <= st


def parse_case(scenario):
    """Parser de respaldo (gramatica) usado como VALIDADOR/fallback."""
    skin = _skin(scenario)
    it, gt = _statement_sentences(scenario)
    init = _parse_facts_grammar(it, skin)
    goal = _parse_facts_grammar(gt, skin)
    objs = sorted({t for f in init | goal for t in f[1:]})
    return skin, init, goal, objs

_RULES = ("Rules (Blocksworld): pickup x needs clear x, ontable x, handempty -> holding x. "
          "putdown x needs holding x -> clear x, ontable x, handempty. "
          "stack x y needs holding x, clear y -> on x y, clear x, handempty. "
          "unstack x y needs on x y, clear x, handempty -> holding x, clear y.\n")


def _facts_str(facts):
    return ", ".join(" ".join(f) for f in sorted(facts))


def build_verify_prompt(init, goal, plan):
    return (_RULES + f"Initial state: {_facts_str(init)}.\nGoal: {_facts_str(goal)}.\n"
            + "Plan: " + "; ".join(" ".join(a) for a in plan) + ".\n"
            + "Is the plan executable step by step and does it reach the goal?\n"
            + "Answer with exactly one word: VALID or INVALID.")


def build_emit_prompt(init, goal, plan, skin):
    cand = "\n".join(to_output(plan, skin))
    return (_RULES + f"Initial state: {_facts_str(init)}.\nGoal: {_facts_str(goal)}.\n"
            + "A candidate optimal plan, verified with a symbolic simulator:\n" + cand
            + "\nWrite the FINAL plan, one action per line, exact format '(verb arg1 arg2)'."
            + " Do not write anything else.")


def to_output(plan, skin):
    tab = _P2MYST if skin == "mystery" else _P2DRON
    return ["(" + " ".join([tab[a[0]]] + list(a[1:])) + ")" for a in plan]


def from_output(text, skin):
    tab = _MYST2P if skin == "mystery" else _DRON2P
    plan = []
    for m in re.finditer(r"\(?\s*([a-z_]+)((?:\s+[a-z]+){1,2})\s*\)?", text.lower()):
        verb, args = m.group(1), m.group(2).split()
        if verb in tab:
            plan.append(tuple([tab[verb]] + args))
    return plan


class AssemblyAgent:
    def __init__(self):
        self.system_prompt = _EXTRACT_SYSTEM
        self.last_audit = None

    def _qwen(self, llm, prompt, max_new_tokens):
        return llm(prompt=prompt, system=self.system_prompt,
                   max_new_tokens=max_new_tokens, temperature=0.0,
                   do_sample=False, enable_thinking=False)

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        audit = {"mode": QWEN_MODE}
        skin = _skin(scenario_context)
        init_txt, goal_txt = _statement_sentences(scenario_context)

        g_init = _parse_facts_grammar(init_txt, skin)
        g_goal = _parse_facts_grammar(goal_txt, skin)
        g_objs = sorted({t for f in g_init | g_goal for t in f[1:]})

        init, goal = g_init, g_goal
        if QWEN_MODE in ("cot_extract", "cot_full"):
            try:
                out = self._qwen(llm_engine_func,
                                 build_extract_prompt(init_txt, goal_txt, skin), 200)
                parsed = parse_llm_facts(out)
                if parsed and _sane(*parsed):
                    init, goal = parsed
                    audit["extract"] = "ok" + ("=grammar" if (init, goal) == (g_init, g_goal)
                                               else "!=grammar")
                else:
                    audit["extract"] = "malformada->fallback gramatica"
            except Exception as e:
                audit["extract"] = f"error:{type(e).__name__}->fallback"

        objs = sorted({t for f in init | goal for t in f[1:]})

        plan = bfs_optimal(init, goal, objs)

        if plan is None or not validate_plan(g_init, g_goal, g_objs, plan):
            audit["replan"] = True
            init, goal, objs = g_init, g_goal, g_objs
            plan = bfs_optimal(init, goal, objs)
        final_plan = plan
        try:
            if QWEN_MODE == "cot_full":
                out = self._qwen(llm_engine_func,
                                 build_emit_prompt(init, goal, plan, skin), 220)
                emitted = from_output(out, skin)
                if emitted and len(emitted) == len(plan) and \
                        validate_plan(g_init, g_goal, g_objs, emitted):
                    final_plan = emitted
                    audit["emit"] = "aceptada"
                else:
                    audit["emit"] = "fallback"
            else:
                out = self._qwen(llm_engine_func,
                                 build_verify_prompt(init, goal, plan), 3)
                audit["verify"] = out.strip()[:20]
        except Exception as e:
            audit["etapaC"] = f"error:{type(e).__name__}"

        audit["plan_len"] = len(final_plan)
        self.last_audit = audit
        if VERBOSE:
            print("   [audit]", audit)
        return to_output(final_plan, skin)
