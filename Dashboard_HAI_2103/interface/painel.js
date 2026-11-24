"use strict";
const el = id => document.getElementById(id);
const inteiro = new Intl.NumberFormat('pt-BR');
const dec = (n,precision=0) => n==null?'—':n.toLocaleString('pt-BR',{minimumFractionDigits:precision,maximumFractionDigits:precision});
const pct = n => n==null?'—':dec(n*100,2)+'%';
const seconds = n => n==null?'Sem alerta':dec(n,Number.isInteger(n)?0:1)+' s';
const reading = new Intl.NumberFormat('pt-BR',{maximumFractionDigits:5});
const two = n => String(n).padStart(2,'0');
const MODELS=['Logística','IF com PCA','Ensemble OU'];
const SENSORS={
 P3_LIT01:'Nível do tanque superior de água',P3_PIT01:'Pressão da água na entrada do tanque superior',
 P1_LIT01:'Nível do tanque de retorno',P1_PIT01:'Pressão na saída do trocador de calor',
 P1_TIT01:'Temperatura na saída do trocador de calor',P1_FT01:'Sinal de vazão do tanque de retorno',
 P2_SIT01:'Rotação da turbina',P4_ST_PT01:'Sinal digital de pressão do vapor',
 P4_ST_TT01:'Sinal digital de temperatura do vapor'
};
// Unidades do manual HAI, seção Data Points. P3_LIT01 permanece na escala
// original do CSV: os valores registrados não correspondem à faixa percentual do manual.
const UNITS={P3_LIT01:'escala original',P3_PIT01:'valor digital',P1_LIT01:'mm',P1_PIT01:'bar',P1_TIT01:'°C',P1_FT01:'mmH₂O',P2_SIT01:'rpm',P4_ST_PT01:'valor digital',P4_ST_TT01:'valor digital'};
const UNIT_AXIS={...UNITS,P3_LIT01:'Escala original',P3_PIT01:'Valor digital',P4_ST_PT01:'Valor digital',P4_ST_TT01:'Valor digital'};
const meta=JSON.parse(el('metadata').textContent);
// As datas do HAI não indicam fuso. Operações UTC preservam exatamente seus rótulos locais.
const base=Date.parse(meta.inicio+'Z');
const iso=i=>new Date(base+i*1000).toISOString();
const clock=i=>iso(i).slice(11,19);
const fullDate=i=>new Intl.DateTimeFormat('pt-BR',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'}).format(base+i*1000);
const state={episode:8,cursor:0,signal:meta.sinais[0],view:'analysis',speed:5,playing:false,timer:0};
const data={signals:[],truth:null,decisions:null,episodes:[],summary:[]};
let geometry=null,framePending=false;
function node(tag,cls,value){const a=document.createElement(tag);if(cls)a.className=cls;if(value!==undefined)a.textContent=value;return a;}
function option(p,value,text){const a=node('option','',text);a.value=value;p.append(a);}
function unpackBytes(text){const raw=atob(text.trim()),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return bytes;}
async function inflate(blob,limit){
 if(typeof DecompressionStream==='undefined')throw Error('Use um navegador recente que aceite DecompressionStream.');
 const reader=blob.stream().pipeThrough(new DecompressionStream('gzip')).getReader();
 const chunks=[];let size=0;
 while(true){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>limit){await reader.cancel();throw Error('Pacote de dados acima do limite previsto.');}chunks.push(value);}
 const result=new Uint8Array(size);let pos=0;for(const x of chunks){result.set(x,pos);pos+=x.byteLength;}return result;
}
const observed=(i,m)=>(data.decisions[i]&(2<<m))!==0;
function median(vals){if(!vals.length)return null;const s=[...vals].sort((a,b)=>a-b),m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2;}
function setEpisodes(){
 const list=[];
 for(let i=0;i<meta.total;i++){
   if(!data.truth[i])continue;
   const start=i;while(i+1<meta.total&&data.truth[i+1])i++;
   list.push({id:list.length+1,start,end:i,n:i-start+1});
 }
 data.episodes=list;
 if(list.length!==12)throw Error('Os rótulos não formam os 12 episódios esperados do test5.');
}
function calculate(){
 const r=MODELS.map(()=>({tp:0,fp:0,fn:0,episodes:0,delay:null}));
 for(const e of data.episodes)e.models=MODELS.map(()=>({n:0,first:null}));
 let eidx=0;
 for(let i=0;i<meta.total;i++){
   while(eidx<data.episodes.length&&i>data.episodes[eidx].end)eidx++;
   const e=data.episodes[eidx],truth=Boolean(data.truth[i]);
   MODELS.forEach((_,m)=>{
     const yes=observed(i,m),s=r[m];
     if(truth&&yes){s.tp++;const x=e.models[m];x.n++;if(x.first==null)x.first=i;}
     else if(truth)s.fn++;
     else if(yes)s.fp++;
   });
 }
 r.forEach((s,m)=>{
   const delays=data.episodes.filter(e=>e.models[m].first!=null).map(e=>e.models[m].first-e.start);
   s.episodes=delays.length;s.delay=median(delays);
   s.precision=s.tp+s.fp?s.tp/(s.tp+s.fp):null;
   s.recall=s.tp/(s.tp+s.fn);
   s.f1=(2*s.tp+ s.fp+s.fn)?2*s.tp/(2*s.tp+s.fp+s.fn):0;
 });
 data.summary=r;
}
function activeEpisode(){return data.episodes[state.episode-1];}
function windowBounds(){const e=activeEpisode();return [Math.max(0,e.start-30),Math.min(meta.total-1,e.end+30)];}
function chartBounds(){return windowBounds();}
function pause(){if(state.timer)clearInterval(state.timer);state.timer=0;state.playing=false;el('play-label').textContent='Reproduzir';}
function seek(i,manual=true){if(manual)pause();const [a,b]=windowBounds();state.cursor=Math.max(a,Math.min(b,Math.round(i)));renderFrame();}
function replay(){
 if(state.playing){pause();return;}
 const [a,b]=windowBounds();if(state.cursor>=b)seek(a);
 const initial=state.cursor,started=performance.now();state.playing=true;el('play-label').textContent='Pausar';
 state.timer=setInterval(()=>{const idx=Math.min(b,initial+Math.floor((performance.now()-started)*state.speed/1000));if(idx!==state.cursor)seek(idx,false);if(idx>=b)pause();},100);
}
function chooseEpisode(id,initial=null){
 pause();const index=Number(id);
 if(!Number.isInteger(index)||index<1||index>data.episodes.length)return;
 state.episode=index;const e=activeEpisode();
 state.cursor=initial==null?e.start:Math.max(Math.max(0,e.start-30),Math.min(Math.min(meta.total-1,e.end+30),initial));
 el('episode-select').value=String(index);
 el('episode-number').textContent='EPISÓDIO '+two(e.id);
 el('episode-sub').textContent=fullDate(e.start)+' · '+clock(e.start)+'–'+clock(e.end)+' · '+inteiro.format(e.n)+' registros de ataque';
 el('previous').disabled=e.id===1;el('next').disabled=e.id===data.episodes.length;
 const [lo,hi]=windowBounds();el('scrub').min=lo;el('scrub').max=hi;
 el('range-min').textContent=clock(lo);el('range-max').textContent=clock(hi);
 renderSide();renderFrame();resizeHost();
}
function renderSide(){
 const e=activeEpisode(),root=el('model-cards');root.replaceChildren();
 MODELS.forEach((name,m)=>{
   const a=e.models[m],delay=a.first==null?null:a.first-e.start;
   const card=node('article','detector-card detector-'+m),head=node('div','detector-head');
   head.append(node('h3','detector-title',name),node('span','detector-tag','Sem alerta agora'));
   const metric=node('div','detector-main');metric.append(node('span','detector-metric-label','Primeiro alerta no episódio'));
   if(a.first!=null){const b=node('button','detector-value detector-link',seconds(delay));b.title='Localizar primeiro alerta';b.setAttribute('aria-label',name+': ir ao primeiro alerta, '+seconds(delay));b.onclick=()=>seek(a.first);metric.append(b);}
   else metric.append(node('strong','detector-value no-alert','Sem alerta'));
   const coverage=node('div','detector-coverage');coverage.append(node('span','','Registros sinalizados no episódio'),node('strong','',inteiro.format(a.n)+' / '+inteiro.format(e.n)));
   const progress=node('div','detector-meter'),fill=node('span','detector-fill');fill.style.width=(100*a.n/e.n)+'%';progress.append(fill);
   progress.setAttribute('role','progressbar');progress.setAttribute('aria-label',name+': '+a.n+' de '+e.n+' registros sinalizados');progress.setAttribute('aria-valuemin','0');progress.setAttribute('aria-valuemax',String(e.n));progress.setAttribute('aria-valuenow',String(a.n));
   card.append(head,metric,coverage,progress);root.append(card);
 });
}
// Altera somente os indicadores ABAIXO do grafico para o segundo selecionado.
// O desenho do grafico, sua legenda e os controles de reproducao permanecem intactos.
function renderInstantAlerts(){
  const i=state.cursor,truth=Boolean(data.truth[i]);
  const flags=MODELS.map((_,m)=>observed(i,m));
  const any=flags.some(Boolean),status=el('comparison-status');
  status.hidden=false;
  status.className='comparison-status '+(truth?(any?'attack-alert':'attack-missed'):(any?'false-alert':'normal-now'));
  status.textContent=truth?(any?'Instante: ataque com alerta':'Instante: ataque sem alerta'):(any?'Instante: falso positivo':'Instante: normal sem alerta');
  el('model-cards').querySelectorAll('.detector-card').forEach((card,m)=>{
    const tp=truth&&flags[m],fp=!truth&&flags[m],miss=truth&&!flags[m];
    card.classList.toggle('alert-now',tp);
    card.classList.toggle('false-now',fp);
    card.classList.toggle('miss-now',miss);
    const tag=card.querySelector('.detector-tag');
    tag.textContent=tp?'Ataque sinalizado agora':fp?'Falso positivo agora':miss?'Ataque sem alerta agora':'Sem alerta agora';
  });
}
function frameText(){
 const i=state.cursor,truth=Boolean(data.truth[i]);
 return (truth?'Registro rotulado como ataque. ':'Registro rotulado como normal. ')+MODELS.map((m,j)=>m+': '+(observed(i,j)?'alerta':'sem alerta')).join('; ')+'.';
}
function renderFrame(){
 const i=state.cursor,e=activeEpisode();el('scrub').value=i;el('scrub').setAttribute('aria-valuetext',clock(i));
 const offset=i<e.start?(e.start-i)+' s antes do início':i>e.end?(i-e.end)+' s após o fim':(i-e.start)+' s desde o início';
 el('clock').textContent=clock(i);
 el('relative-time').textContent=offset;
 el('signal-desc').textContent=(SENSORS[state.signal]||state.signal)+' · '+state.signal+' · leitura '+reading.format(data.signals[meta.sinais.indexOf(state.signal)][i])+' '+(UNITS[state.signal]||'escala original');
 renderInstantAlerts();
 drawTimeline();
}

function drawTimeline() {
  if(state.view!=='analysis')return;
  const canvas=el('timeline'), rect=canvas.getBoundingClientRect();if(rect.width<60||rect.height<100)return;
  const w=rect.width, h=rect.height, compact=w<520;
  const ratio=Math.min(3,window.devicePixelRatio||1);
  if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
  const [a,b]=chartBounds(), left=compact?91:120, right=w-(compact?15:24), width=right-left;
  const x=i=>left+(i-a)/(b-a)*width;
  const top=34, baseline=h*.51, bottom=h-32, bandTop=h*.63, bandH=(bottom-bandTop)/3;
  const array=data.signals[meta.sinais.indexOf(state.signal)];
  let min=Infinity,max=-Infinity;
  for(let i=a;i<=b;i++){min=Math.min(min,array[i]);max=Math.max(max,array[i]);}
  const pad=(max-min)*.14 || Math.max(Math.abs(max)*.005,.5); min-=pad;max+=pad;
  const y=v=>baseline-(v-min)/(max-min)*(baseline-top);
  const txt=(s,xx,yy,color='#8fa4bb',size=11,align='left',bold=false)=>{
    ctx.fillStyle=color;ctx.font=(bold?'600 ':'')+size+'px "Segoe UI", system-ui, sans-serif';ctx.textAlign=align;ctx.fillText(s,xx,yy);
  };
  const line=(x1,y1,x2,y2,color,ww=1,dash=[])=>{ctx.beginPath();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=ww;ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);};
  let k=a;while(k<=b){if(!data.truth[k]){k++;continue;}const begin=k;while(k<=b&&data.truth[k])k++;
    const x1=Math.max(left,x(begin-.5)),x2=Math.min(right,x(k-.5));ctx.fillStyle='rgba(245,166,74,.13)';ctx.fillRect(x1,top,x2-x1,bottom-top);
    if(begin>a)line(x1,top,x1,bottom,'#9a7441',1,[3,4]);
    if(k<=b)line(x2,top,x2,bottom,'#9a7441',1,[3,4]);
  }
  const tickFormatter=new Intl.NumberFormat('pt-BR',{maximumFractionDigits:(max-min)<.1?4:(max-min)<10?2:(max-min)<120?1:0});
  for(let j=0;j<3;j++){
    const v=min+(max-min)*(.12+j*.38), yy=y(v);
    line(left,yy,right,yy,'#26364b',.8);txt(tickFormatter.format(v),left-7,yy+3.5,'#c1d2e4',compact?12:14,'right');
  }
  const intervals=compact?3:5;
  for(let j=0;j<=intervals;j++){
    const i=Math.round(a+(b-a)*j/intervals),xx=x(i);
    line(xx,top,xx,bottom,'#26364b',.65,[2,4]);
    txt(clock(i),xx,bottom+18,'#c1d2e4',compact?11:13,j===intervals?'right':j===0?'left':'center');
  }
  ctx.save();ctx.beginPath();ctx.rect(left,top,width,baseline-top+2);ctx.clip();
  ctx.lineWidth=compact?1.6:1.8;ctx.lineJoin='round';ctx.strokeStyle='#54a9ff';ctx.beginPath();
  for(let i=a;i<=b;i++){if(i===a)ctx.moveTo(x(i),y(array[i]));else ctx.lineTo(x(i),y(array[i]));}ctx.stroke();ctx.restore();
  txt('Alertas',left-7,bandTop-3,'#d0deec',compact?12:14,'right',true);
  MODELS.forEach((label,m)=>{
    const cy=bandTop+bandH*(m+.5);
    txt(label,left-7,cy+3,'#d0deec',compact?12:14,'right');
    line(left,cy,right,cy,'#26364b',.85);
    for(let i=a;i<=b;i++)if(observed(i,m)) {
      const xx=x(i), radius=compact?2.8:3.6;
      ctx.fillStyle=data.truth[i]?'#36c9a6':'#fa8a71';ctx.beginPath();
      if(data.truth[i])ctx.arc(xx,cy,radius,0,2*Math.PI);
      else {ctx.moveTo(xx,cy-radius-1);ctx.lineTo(xx+radius+1,cy);ctx.lineTo(xx,cy+radius+1);ctx.lineTo(xx-radius-1,cy);ctx.closePath();}
      ctx.fill();
    }
  });
  const xx=x(state.cursor), yy=y(array[state.cursor]);
  line(xx,top-3,xx,bottom,'#b8caff',1.4,[4,3]);ctx.beginPath();ctx.arc(xx,yy,4,0,2*Math.PI);ctx.fillStyle='#111f31';ctx.fill();ctx.strokeStyle='#54a9ff';ctx.lineWidth=2;ctx.stroke();
  const label=clock(state.cursor), lw=84,lx=Math.min(Math.max(xx-lw/2,left),right-lw);
  ctx.fillStyle='#244c79';ctx.fillRect(lx,4,lw,23);txt(label,lx+lw/2,20,'white',14,'center',true);
  txt(UNIT_AXIS[state.signal]||'Leitura',left-7,top-8,'#d0deec',compact?11:13,'right',true);
  canvas.setAttribute('aria-label',SENSORS[state.signal]+', '+clock(state.cursor)+', valor '+dec(array[state.cursor],3)+'. '+frameText());
  geometry={a,b,left,right};
}


function renderResults(){
 const attacks=data.episodes.reduce((n,e)=>n+e.n,0),r=data.summary,lr=r[0],ou=r[2];
 el('results-sub').textContent=inteiro.format(meta.total)+' registros · '+inteiro.format(attacks)+' de ataque · '+data.episodes.length+' episódios';
 const root=el('metrics-body');root.replaceChildren();
 MODELS.forEach((name,j)=>{
   const s=r[j],row=node('tr');
   [name,dec(s.f1,4),inteiro.format(s.tp),inteiro.format(s.fp),s.episodes+' / '+data.episodes.length,seconds(s.delay)].forEach((v,i)=>{
     const cell=node(i===0?'th':'td','',v);if(i===0)cell.scope='row';row.append(cell);
   });root.append(row);
 });
 const moreTP=ou.tp-lr.tp,moreFP=ou.fp-lr.fp;
 el('result-summary').textContent='Na execução incluída, o ensemble OU sinalizou '+inteiro.format(moreTP)+' registros de ataque a mais do que a Regressão Logística, com '+inteiro.format(moreFP)+' falsos positivos adicionais. O F1 passou de '+dec(lr.f1,4)+' para '+dec(ou.f1,4)+'.';
 // Dados e avaliações devem ser coerentes com o artigo antes da defesa.
 el('consistency-warning').hidden=!(lr.tp!==1116||lr.fp!==643||ou.tp!==1260||ou.fp!==1118);
}
function switchView(key){
 pause();state.view=key;
 el('analysis-view').hidden=key!=='analysis';el('results-view').hidden=key!=='results';
 for(const [id,v] of [['tab-analysis','analysis'],['tab-results','results']]){const b=el(id);b.classList.toggle('active',key===v);if(key===v)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');}
 if(key==='analysis')renderFrame();else renderResults();
 resizeHost();
}
function bind(){
 data.episodes.forEach(e=>option(el('episode-select'),e.id,'Episódio '+two(e.id)));
 meta.sinais.forEach(s=>option(el('signal'),s,s+' · '+(UNITS[s]||'escala original')));
 el('episode-select').onchange=e=>chooseEpisode(e.target.value);
 el('previous').onclick=()=>chooseEpisode(state.episode-1);el('next').onclick=()=>chooseEpisode(state.episode+1);
 el('signal').onchange=e=>{state.signal=e.target.value;renderFrame();};
 el('play').onclick=replay;el('restart').onclick=()=>seek(windowBounds()[0]);
 el('speed').onchange=e=>{const again=state.playing;pause();state.speed=Number(e.target.value);if(again)replay();};
 el('scrub').oninput=e=>seek(Number(e.target.value));
 el('timeline').onclick=e=>{if(!geometry)return;const p=e.clientX-el('timeline').getBoundingClientRect().left,{left,right,a,b}=geometry;if(p>=left&&p<=right)seek(a+(p-left)/(right-left)*(b-a));};
 el('tab-analysis').onclick=()=>switchView('analysis');el('tab-results').onclick=()=>switchView('results');
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
 const ro=new ResizeObserver(()=>{if(state.view==='analysis')drawTimeline();resizeHost();});ro.observe(el('main'));ro.observe(el('timeline'));
}
let streamlitHost=false,reportedHeight=0;
function postHost(type,values){if(window.parent!==window)window.parent.postMessage(Object.assign({isStreamlitMessage:true,type},values),'*');}
function resizeHost(){
 if(!streamlitHost||framePending)return;
 framePending=true;
 requestAnimationFrame(()=>{
   framePending=false;
   // Use actual content, not the iframe's previous viewport height.
   // This allows a shorter Results tab to reduce the frame height.
   const visible=['.topbar','#loading','#main','#footer']
     .map(selector=>document.querySelector(selector))
     .filter(element=>element&&!element.hidden&&getComputedStyle(element).display!=='none');
   const bottom=visible.reduce((max,element)=>Math.max(max,element.getBoundingClientRect().bottom+window.scrollY),0);
   const height=Math.max(1,Math.ceil(bottom)+2);
   if(height!==reportedHeight){
     postHost('streamlit:setFrameHeight',{height});
     reportedHeight=height;
   }
 });
}
window.addEventListener('message',e=>{if(e.source===window.parent&&e.data?.type==='streamlit:render'){streamlitHost=true;resizeHost();}});
if(window.parent!==window)postHost('streamlit:componentReady',{apiVersion:1});
async function start(){
 try {
   const raw=unpackBytes(el('measurements').textContent),stride=meta.total*8;
   const bytes=await inflate(new Blob([raw]),stride*meta.sinais.length+meta.total);
   if(bytes.byteLength!==stride*meta.sinais.length+meta.total)throw Error('Pacote de registros incompleto.');
   data.signals=meta.sinais.map((_,i)=>new Float64Array(bytes.buffer,i*stride,meta.total));
   data.decisions=bytes.subarray(stride*meta.sinais.length);data.truth=Uint8Array.from(data.decisions,n=>n&1);
   setEpisodes();calculate();bind();
   el('fact-total').textContent=inteiro.format(meta.total);
   el('fact-attacks').textContent=inteiro.format(data.episodes.reduce((n,e)=>n+e.n,0));
   el('fact-episodes').textContent=String(data.episodes.length);
   el('loading').hidden=true;el('main').hidden=false;el('footer').hidden=false;
   const e=data.episodes[7];chooseEpisode(8,e.models[2].first??e.start);
   if(document.fonts)document.fonts.ready.then(()=>{drawTimeline();resizeHost();});
 }catch(err){el('loading').hidden=false;el('loading').textContent='Não foi possível carregar o painel: '+err.message;resizeHost();}
}
start();
