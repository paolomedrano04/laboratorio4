"""Prueba SIN GPU: reemplaza qwen por un stub para validar el pipeline
(parser, BFS, prompts, parseo de salida, evaluador). La entrega real debe
generarse en Colab con el Qwen real (submit.py)."""
import json, sys, time
sys.path.insert(0, '.')
import student_agent
from student_agent import AssemblyAgent
from evaluator import calcular_score_plan

def qwen_stub_verify(prompt, system=None, **kw):
    return "VALID"

def qwen_stub_emit(prompt, system=None, **kw):
    # simula a Qwen copiando el plan candidato mostrado en el prompt (temp 0)
    lines = [l for l in prompt.splitlines() if l.startswith("(")]
    return "\n".join(lines)

def run(mode, stub, examples):
    student_agent.QWEN_MODE = mode
    ag = AssemblyAgent()
    tot = 0.0
    t0 = time.time()
    for e in examples:
        plan = ag.solve(e['scenario_context'], stub)
        tot += calcular_score_plan(plan, e['target_action_sequence'])
    dt = time.time() - t0
    print(f"[{mode:6s}] score medio = {tot/len(examples):.3f}/10  "
          f"({len(examples)} casos, {dt:.2f}s de logica sin LLM)")

if __name__ == "__main__":
    ex = json.load(open("Examples.json"))
    run("verify", qwen_stub_verify, ex)
    run("emit",   qwen_stub_emit,   ex)
    # y sobre Task.json: solo comprobar que parsea y produce salida bien formada
    tasks = json.load(open("Task.json"))
    student_agent.QWEN_MODE = "verify"
    ag = AssemblyAgent()
    lens = []
    for t in tasks:
        p = ag.solve(t['scenario_context'], qwen_stub_verify)
        assert all(a.startswith("(") and a.endswith(")") for a in p)
        lens.append(len(p))
    print(f"[tasks ] 50/50 resueltos, longitudes: min={min(lens)} max={max(lens)}")
