'use strict';
const $ = q => document.querySelector(q);
let recipes = {}, plan = null, busy = false;
function node(tag, text, cls) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }
async function api(path, value) {
  const response = await fetch(path, value === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-StrokeWeave-Request':'local-research-v1'},body:JSON.stringify(value)});
  const body = await response.json(); if (!response.ok) throw new Error(body.error || 'Request unavailable'); return body;
}
function renderJobs(jobs) {
  const count = states => jobs.filter(j => states.includes(j.state)).length;
  $('#count-all').textContent = jobs.length; $('#count-complete').textContent = count(['COMPLETE']);
  $('#count-review').textContent = count(['REFER','ERROR']); $('#count-active').textContent = count(['QUEUED','RUNNING']);
  const table = $('#jobs'); table.replaceChildren(); $('#empty').hidden = jobs.length > 0;
  for (const job of jobs) {
    const tr = node('tr'), title = node('td', recipes[job.recipe] || job.recipe); title.append(node('small',job.id.slice(0,12))); tr.append(title);
    const status = node('td'); status.append(node('span',job.state,'status '+job.state)); tr.append(status);
    tr.append(node('td',`${job.attempts} / ${job.attempt_budget}`),node('td',job.state === 'COMPLETE' && job.result ? job.result.volume_mL.toFixed(3)+' mL' : 'No accepted result'));
    const cell = node('td'), button = node('button','Audit ↗','quiet'); button.onclick = () => openAudit(job.id); cell.append(button); tr.append(cell); table.append(tr);
  }
}
async function refresh() {
  if (busy) return; busy = true;
  try { const [health, listing] = await Promise.all([api('/api/health'),api('/api/jobs')]);
    if (health.patient_execution !== false || health.training_enabled !== false) throw new Error('Unexpected boundary');
    renderJobs(listing.jobs); $('#connection').textContent = health.worker_running ? 'Local · connected' : 'Worker stopped · review required'; $('#connection').className = 'badge online'; $('#queue-message').textContent = '';
  } catch (error) { $('#connection').textContent = 'Connection unavailable'; $('#connection').className='badge'; $('#queue-message').textContent = error.message; }
  finally { busy = false; }
}
async function openAudit(id) {
  try { const record = await api('/api/jobs/'+id); $('#detail-title').textContent=(recipes[record.recipe] || record.recipe)+' · '+record.state;
    $('#detail-json').textContent=JSON.stringify(record,null,2); $('#detail-warning').textContent='Generated technical fixture only. A result is not a clinical finding; hash checks are not authenticated signatures.';
    const links = $('#export-links'); links.replaceChildren();
    for (const [suffix,label] of [['report.json','JSON report ↗'],['report.html','HTML report ↗'],['export','Audit bundle ↓']]) { const a=node('a',label); a.href='/api/jobs/'+id+'/'+suffix; a.download=''; links.append(a); }
    $('#detail').showModal();
  } catch(error) { $('#queue-message').textContent=error.message; }
}
$('#close-detail').onclick=()=>$('#detail').close(); $('#refresh').onclick=refresh;
$('#volume-example').onclick=()=>{$('#task-text').value='Measure the volume of the generated spatial phantom.';};
$('#preview').onclick=async()=>{ const button=$('#preview'); button.disabled=true; $('#execute').hidden=true; plan=null;
  try { const value=await api('/api/plans',{text:$('#task-text').value}); plan=value; $('#plan').hidden=false; $('#plan-status').textContent=value.status.replaceAll('_',' '); $('#plan-reason').textContent=value.explanation;
    const list=$('#plan-steps'); list.replaceChildren(); for(const step of value.tasks) list.append(node('li',recipes[step.recipe] || step.recipe)); $('#execute').hidden=value.status!=='READY'; $('#task-message').textContent='Preview created. No task has run.';
  } catch(error) { $('#task-message').textContent=error.message; } finally { button.disabled=false; }
};
$('#execute').onclick=async()=>{ if(!plan || plan.status!=='READY')return; const button=$('#execute'); button.disabled=true;
  if(!plan.request_key)plan.request_key='web_'+crypto.randomUUID().replaceAll('-','');
  try { const result=await api('/api/batches',{request_key:plan.request_key,tasks:plan.tasks}); $('#task-message').textContent=`${result.job_ids.length} generated task(s) queued. Original task text is not persisted.`; button.hidden=true; await refresh(); }
  catch(error) { $('#task-message').textContent=error.message; } finally { button.disabled=false; }
};
async function init() { try { const data=await api('/api/tools'); recipes=data.recipes; const list=$('#tools'); list.replaceChildren(); for(const tool of data.tools) { const card=node('article',undefined,'tool-card'); card.append(node('span',tool.enabled?'GENERATED TOOL ENABLED':'EXTERNAL PLUGIN DISABLED','tool-status'),node('h3',tool.id),node('p',tool.scope),node('small',tool.training_exposure)); list.append(card); }} catch(error) { $('#task-message').textContent=error.message; } await refresh(); }
init(); setInterval(()=>{if(!document.hidden)refresh();},3000);
