"""Local demo viewer; all model/example text is rendered with textContent."""

HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grand Mobile · Support Copilot</title><style>
:root{color-scheme:dark;--bg:#10131b;--panel:#191f2c;--line:#313a50;--muted:#b6c2d8;--accent:#a5baff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2ff;font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px 70px}
h1{font-size:clamp(27px,4vw,42px);line-height:1.2}h2{font-size:23px;margin-top:36px}a{color:var(--accent)}p{margin:10px 0}.muted,small{color:var(--muted)}
.banner,.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;margin:14px 0}.banner{border-left:4px solid var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}.metric strong{display:block;font-size:30px}.metric{margin:0}
label{display:block;margin:14px 0 6px}textarea,select,button{font:inherit;border:1px solid #56627e;border-radius:9px}textarea,select{color:inherit;background:#101521;padding:10px;width:100%}
textarea{min-height:135px;resize:vertical}button{background:#b4c6ff;color:#10131b;padding:10px 16px;cursor:pointer;font-weight:650}button.secondary{background:var(--panel);color:#edf2ff}
button:disabled{opacity:.55;cursor:wait}button:focus-visible,a:focus-visible,textarea:focus-visible,select:focus-visible{outline:3px solid #f4c35b;outline-offset:3px}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.row>div{flex:1;min-width:180px}
.answer,.question{white-space:pre-wrap;overflow-wrap:anywhere}.tag{display:inline-block;padding:2px 9px;border:1px solid #53607b;border-radius:7px;margin:4px 6px 4px 0;font-size:14px}.error{color:#ffb9b9}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;text-align:left}caption{text-align:left;color:var(--muted);padding:0 0 12px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:650}td{font-variant-numeric:tabular-nums}th:first-child{min-width:175px}
</style><script src="/assets/dashboard.js" defer></script></head><body><main>
<div class="row"><span class="tag">Локальный сервис</span><a href="/docs">Документация API</a><a href="/v1/meta">Метаданные</a></div>
<h1>Grand Mobile<br>Support Copilot</h1><p class="muted">Проверить обращение, увидеть источники и сравнить фактические ответы.</p>
<div class="banner">Все ответы — черновики для сотрудника. Платежи, обжалования и доступ к аккаунту требуют человека. Изображения не анализируются.</div>
<p id="health" role="status" aria-live="polite">Проверяем готовность…</p>
<h2>Попробовать обращение</h2><form id="form" class="card"><div class="row"><div><label for="kind">Канал</label><select id="kind"><option value="complaint">Жалоба на игрока</option><option value="report">Внутриигровой репорт</option></select></div>
<div><label for="lang">Подсказка языка сервера</label><select id="lang"><option value="">Определить автоматически</option><option value="ru">Русский</option><option value="kk">Қазақша</option><option value="uz">O‘zbekcha</option><option value="en">English</option></select></div></div>
<label for="question">Текст игрока</label><textarea id="question" maxlength="4000" required>Игрок оскорбил меня в чате, сервер 12.</textarea>
<div class="row"><button type="submit">Получить черновик</button><button type="button" class="secondary" id="classify">Только классификация</button><button type="button" class="secondary" id="search">Найти правила</button></div>
<p id="status" role="status" aria-live="polite"></p><div id="result"></div></form>
<h2>Текущая проверка качества · до и после</h2><p class="muted">Синтетические сценарии, не человеческая оценка и не точность на реальных обращениях. Проверка источников не доказывает правильность смысла.</p>
<div id="qualityStatus" role="status" aria-live="polite">Загружаем новую проверку…</div><div id="qualityMetrics"></div>
<h2>Исторический прогон · 36 сценариев</h2><p class="muted">Архивные результаты и примеры до текущих исправлений. Это не текущая оценка качества. Неудачные запросы включены в расчёт.</p>
<button class="secondary" id="reload">Обновить метрики</button><div class="grid" id="metrics"></div><div id="classifierMetrics" class="card"></div>
<h2>Архив · языки и сложный ввод</h2><p class="muted">Прежние синтетические проверки: RU / KK / UZ / EN, смешанные тексты, опечатки, legacy KK, неверные пункты и отсутствие изображений. und означает, что нужно уточнить язык.</p><div class="grid" id="inputMetrics"></div>
<h2>Архивные ответы из прежнего запуска</h2><p class="muted">Примеры включают успешные случаи и ошибки прежнего набора, а не ответы текущей версии.</p><div id="examples"></div>
</main></body></html>'''

JS = r'''
"use strict";
const el = id => document.getElementById(id);
function node(tag, text, cls) {const e=document.createElement(tag); if(text!==undefined)e.textContent=String(text); if(cls)e.className=cls; return e;}
function tag(text){return node('span',text,'tag');}
function sources(parent, ids){for(const id of ids||[]){const m=/^rules:(\d+):(\d+(?:\.\d+)+)$/.exec(id);if(!m)continue;const a=node('a',id,'tag');a.href='/v1/rag/rules/'+m[1]+'/'+m[2];a.target='_blank';a.rel='noopener';parent.append(a);}}
async function request(path, payload){const response=await fetch(path,payload?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{});const data=await response.json();if(!response.ok){const err=new Error(data.error?.code==='busy'?'Сервис занят, повторите запрос через несколько секунд.':data.error?.message||'Запрос не выполнен.');throw err;}return data;}
function metric(label,value){const d=node('div',undefined,'card metric');d.append(node('small',label),node('strong',value));return d;}
function pct(value){return typeof value==='number'?(value*100).toFixed(1)+'%':'нет данных';}
function validQualityGroup(g){
 if(!g||typeof g!=='object')return false;
 const counts=['cases','category_correct','category_cases','required_citation_hits','required_citation_cases','pseudo_citation_cases','intake_template_cases','fallback_cases'];
 const ratios=['category_macro_f1','priority_macro_f1','p0_recall'];
 return counts.every(k=>Number.isSafeInteger(g[k])&&g[k]>=0)&&ratios.every(k=>Number.isFinite(g[k])&&g[k]>=0&&g[k]<=1)&&
 ['latency_p50_seconds','latency_p95_seconds'].every(k=>Number.isFinite(g[k])&&g[k]>=0)&&g.category_correct<=g.category_cases&&
 g.required_citation_hits<=g.required_citation_cases&&g.required_citation_cases<=g.cases&&g.pseudo_citation_cases<=g.cases&&g.intake_template_cases<=g.cases&&g.fallback_cases<=g.cases;
}
function validQualityStress(g){
 return g&&typeof g==='object'&&['cases','required_citation_hits','required_citation_cases','pseudo_citation_cases','critical_correct','critical_cases','language_matches'].every(k=>Number.isSafeInteger(g[k])&&g[k]>=0)&&
 g.required_citation_hits<=g.required_citation_cases&&g.required_citation_cases<=g.cases&&g.pseudo_citation_cases<=g.cases&&g.critical_correct<=g.critical_cases&&g.critical_cases<=g.cases&&g.language_matches<=g.cases;
}
function qualityTable(before,after){
 const wrap=node('div',undefined,'card table-scroll'),table=node('table');table.append(node('caption','Наблюдаемые результаты одного последовательного HTTP-прогона; все попытки учтены.'));
 const head=node('thead'),hr=node('tr');for(const label of ['Показатель','До','После']){const h=node('th',label);h.scope='col';hr.append(h);}head.append(hr);table.append(head);
 const body=node('tbody');const specs=[
 ['Случаев',g=>String(g.cases)],['Категория',g=>g.category_correct+' / '+g.category_cases],
 ['Категории · macro-F1',g=>g.category_macro_f1.toFixed(3)],['Приоритеты · macro-F1',g=>g.priority_macro_f1.toFixed(3)],
 ['Критичные p0 · recall',g=>pct(g.p0_recall)],['Нужная ссылка в ответе',g=>g.required_citation_hits+' / '+g.required_citation_cases],
 ['Ответы с псевдоссылками',g=>String(g.pseudo_citation_cases)],['Сбор сведений без LLM',g=>String(g.intake_template_cases)],
 ['Отклонённый черновик → шаблон',g=>String(g.fallback_cases)],['Время · p50 / p95',g=>g.latency_p50_seconds.toFixed(2)+' / '+g.latency_p95_seconds.toFixed(2)+' с']];
 for(const [label,format]of specs){const tr=node('tr'),h=node('th',label);h.scope='row';tr.append(h,node('td',format(before)),node('td',format(after)));body.append(tr);}table.append(body);wrap.append(table);return wrap;
}
async function loadQuality(){
 const status=el('qualityStatus'),target=el('qualityMetrics');status.textContent='Загружаем новую проверку…';target.replaceChildren();
 try{const response=await fetch('/v1/quality');if(response.status===404){status.textContent='Новая проверка ещё не завершена';return;}if(!response.ok)throw new Error('unavailable');
 const r=await response.json();if(r.schema_version!==1||r.human_gold!==false||r.label_kind!=='authored_synthetic_quality_comparison_not_human_gold'||!validQualityGroup(r.comparison?.before)||!validQualityGroup(r.comparison?.after))throw new Error('invalid');
 if(r.extra_stress!=null&&!validQualityStress(r.extra_stress))throw new Error('invalid');
 status.textContent='Проверка завершена. Человеческая оценка качества ещё не проведена.';target.append(qualityTable(r.comparison.before,r.comparison.after));
 target.append(node('p','Часть запросов обрабатывается явным шаблоном сбора сведений без LLM. Это изменение маршрутизации и защиты, а не доказательство улучшения весов модели.','muted'));
 if(r.extra_stress){const g=r.extra_stress;const refs=g.required_citation_cases?g.required_citation_hits+' / '+g.required_citation_cases:'не требуются';target.append(node('p','Отдельные новые стрессовые сценарии: '+g.cases+' · нужные ссылки '+refs+' · псевдоссылки '+g.pseudo_citation_cases+' · критичные темы '+g.critical_correct+' / '+g.critical_cases+' · язык '+g.language_matches+' / '+g.cases+'.','muted'));}
 const notes={synthetic_not_human_gold:'Сценарии синтетические, человеческих эталонов нет.',no_native_language_review:'Проверка носителями языков ещё не проведена.',templates_are_not_model_training:'Шаблоны и защитные правила не означают дообучение модели.',same_cases_used_for_development:'Сравниваемые сценарии использовались при исправлениях.',semantic_accuracy_unmeasured:'Смысловая точность ответов не измерена людьми.',single_sequential_run_not_SLO:'Один последовательный прогон не подтверждает производственный SLA.'};
 if(Array.isArray(r.limitations)){const list=node('ul');for(const note of r.limitations.slice(0,8)){if(typeof note==='string')list.append(node('li',notes[note]||note.slice(0,800)));}target.append(list);}
 }catch(e){target.replaceChildren();status.textContent='Новая проверка качества недоступна. Обращение можно проверить через форму выше.';}
}
async function loadMetrics(){
 const currentQuality=loadQuality();
 try {const r=await request('/v1/robustness');const m=r.summary.overall;el('inputMetrics').replaceChildren(metric('Определение языка',m.primary_language_matches+' / '+m.cases),metric('Завершены генерации',m.full_generations+' / '+m.expected_generated_cases),metric('Уточнение und',m.clarification_route_matches+' / '+m.expected_clarification_cases),metric('p95 генерации',m.latency_expected_generation.p95_seconds.toFixed(2)+' с'));}catch(e){el('inputMetrics').replaceChildren(node('p',e.message,'muted'));}
 try {const r=await request('/v1/evaluation');const m=r.metrics.overall;el('metrics').replaceChildren(
 metric('Полный вывод модели',m.full_generations+' / '+m.cases),metric('Обязательные ссылки',m.required_effective_citations_satisfied+' / '+m.citation_required_cases),
 metric('Нужный пункт найден · top10',m.retrieval.required_source_in_top10+' / '+m.retrieval.citation_required_cases),metric('p95 · малый набор',m.latency.p95_seconds.toFixed(2)+' с'));
 el('examples').replaceChildren();for(const x of r.examples){const c=node('article',undefined,'card');c.append(tag(x.language.toUpperCase()),tag(x.category||'нет категории'),tag(x.priority||'нет приоритета'));
 if(x.fallback)c.append(tag('Резервный шаблон'));c.append(node('p',x.question,'question'),node('p',x.answer,'answer'));sources(c,x.citations);c.append(node('small','Время: '+x.elapsed_seconds.toFixed(2)+' с · Требуется сотрудник'));
 if(x.violations.length){const details=node('details');details.append(node('summary','Какие проверки не пройдены'),node('pre',x.violations.join('\n')));c.append(details);}el('examples').append(c);}
 }catch(e){el('metrics').replaceChildren(node('p',e.message,'error'));}
 try{const c=await request('/v1/classifier/metrics');const w=c.weak_label_holdout_agreement,s=c.authored_synthetic_challenge;el('classifierMetrics').replaceChildren(node('h3','Классификатор · E5 + линейные головы'),
 node('p','Согласие с автоматической разметкой: '+pct(w.category.accuracy)+' · n='+w.category.n),node('p','Macro-F1 приоритета на слабой разметке: '+w.priority.macro_f1_supported_classes.toFixed(3)),
 node('p','Отдельные синтетические категории: '+pct(s.category.accuracy)+' · n='+s.category.n),node('small','Это метрики исходной обученной головы. В сервисе при неуверенности действуют резервные правила и независимый контроль приоритета. Человеческой разметки нет.'));
 const h=c.integrated_challenge;if(h){el('classifierMetrics').append(node('p','Сервис с резервными правилами: категории '+pct(h.models.integrated.category.accuracy)+' против '+pct(h.models.baseline.category.accuracy)+' у одних правил · n='+h.n),node('p','Macro-F1 приоритета сервиса: '+h.models.integrated.priority.macro_f1_supported_classes.toFixed(3)+' · неуверенная модель: '+h.learned_abstentions+' / '+h.n));}
 }catch(e){el('classifierMetrics').replaceChildren(node('p',e.message,'error'));}
 await currentQuality;
}
function payload(){const p={kind:el('kind').value,messages:[{role:'player',text:el('question').value}]};if(el('lang').value)p.server_lang=el('lang').value;return p;}
async function run(mode){if(!el('form').reportValidity())return;const buttons=el('form').querySelectorAll('button');buttons.forEach(b=>b.disabled=true);el('status').textContent='Выполняется…';el('result').replaceChildren();const start=performance.now();
 try{const p=payload();const data=await request(mode==='search'?'/v1/rag/search':mode==='classify'?'/v1/classify':'/triage',mode==='search'?{query:p.messages[0].text,lang:p.server_lang||null,limit:5}:p);
 const target=el('result');if(mode==='search'){for(const hit of data.results){const d=node('div',undefined,'card');sources(d,[hit.canonical_id]);d.append(node('p',hit.text,'answer'));target.append(d);}if(!data.results.length)target.append(node('p','Подходящее правило не подтверждено: уточните обстоятельства.'));}
 else {target.append(tag(data.lang),tag(data.category),tag(data.priority),tag('Требуется сотрудник'));if(data.draft_reply){target.append(node('p',data.draft_reply.text,'answer'));sources(target,data.draft_reply.cited);}target.append(node('p','Оценка классификатора: '+pct(data.confidence)+' — не вероятность правильного решения.','muted'));}
 el('status').textContent='Готово за '+((performance.now()-start)/1000).toFixed(2)+' с.';
 }catch(e){el('status').textContent=e.message;}finally{buttons.forEach(b=>b.disabled=false);}}
el('form').addEventListener('submit',e=>{e.preventDefault();run('triage');});el('classify').addEventListener('click',()=>run('classify'));el('search').addEventListener('click',()=>run('search'));el('reload').addEventListener('click',loadMetrics);
request('/ready').then(h=>el('health').textContent=h.ready?'Сервис готов · модель и база знаний загружены':'Сервис не готов').catch(e=>el('health').textContent=e.message);loadMetrics();
'''
