"""Self-contained dashboard HTML page — Mashgin Line Monitor (dark).

Deep-navy canvas, electric-blue accent, Space Grotesk instrument numerals. A
status-reactive hero (live in-line count + sparkline) is the centerpiece; the
checkout-speed strip surfaces the Mashgin-vs-staffed multiplier. Served by
``server.py``; fed by the ``/metrics`` JSON payload from ``metrics.py``.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mashgin · Line Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root{
  --canvas:#0a0e17; --paper:#121826; --sunken:#0e1421; --ink:#eaeef7;
  --brand:#5878ff; --brand-press:#8aa0ff; --brand-wash:#19224180;
  --muted:#8b94a9; --hair:#232c3f; --hair2:#1a2233; --slate:#7c8699; --slate-wash:#1a2130;
  --display:"Space Grotesk",system-ui,sans-serif;
  --ui:"Inter",system-ui,sans-serif;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.4);
  --shadow-lg:0 2px 6px rgba(0,0,0,.5),0 22px 56px rgba(0,0,0,.5);
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--canvas);color:var(--ink);
  font-family:var(--ui);-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:1160px;margin:0 auto;padding:34px 28px 84px}
.eyebrow{font-size:.7rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}

/* header */
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:30px}
.mark{display:flex;align-items:baseline;gap:11px}
.logo{font-family:var(--display);font-weight:700;font-size:1.32rem;letter-spacing:-.025em;color:var(--ink)}
.logo b{color:var(--brand)}
.mark .div{width:1px;height:17px;background:var(--hair);align-self:center}
.mark .sub{font-size:.82rem;font-weight:600;color:var(--muted);letter-spacing:.01em}
.live{display:flex;align-items:center;gap:9px;font-family:var(--display);
  font-weight:500;font-size:.86rem;color:var(--ink);font-variant-numeric:tabular-nums}
.live .lbl{font-family:var(--ui);font-size:.68rem;font-weight:600;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 0 0 rgba(52,211,153,.5);
  animation:pulse 2.4s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.45)}70%{box-shadow:0 0 0 7px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}

/* hero band */
.hero{display:grid;grid-template-columns:1.7fr 1fr 1fr;gap:16px;margin-bottom:16px}
.tile{position:relative;background:var(--paper);border:1px solid var(--hair);
  border-radius:18px;box-shadow:var(--shadow);padding:22px 24px}
.tile.primary{padding-left:30px;box-shadow:var(--shadow-lg);overflow:hidden}
.tile.primary::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;
  background:var(--muted);transition:background .5s}
.tile.primary.low::before {background:#34d399}
.tile.primary.med::before {background:#fbbf24}
.tile.primary.high::before{background:#fb7185}
.tile-top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.chip{font-family:var(--ui);font-weight:700;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  padding:5px 11px;border-radius:999px;border:1px solid;color:var(--muted);
  background:var(--hair2);border-color:var(--hair)}
.chip.low {color:#34d399;background:rgba(52,211,153,.12);border-color:rgba(52,211,153,.3)}
.chip.med {color:#fbbf24;background:rgba(251,191,36,.12);border-color:rgba(251,191,36,.3)}
.chip.high{color:#fb7185;background:rgba(251,113,133,.12);border-color:rgba(251,113,133,.32)}
.bignum{font-family:var(--display);font-weight:600;letter-spacing:-.035em;line-height:.92;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.primary .bignum{font-size:clamp(3.4rem,9vw,5.6rem)}
.tile.compact .bignum{font-size:clamp(2.4rem,5vw,3rem)}
.rowend{display:flex;align-items:flex-end;gap:12px;margin-top:10px}
.roller{display:flex}
.unit{font-family:var(--ui);font-weight:500;font-size:.95rem;color:var(--muted);padding-bottom:.55rem}
.roll-col{height:1em;overflow:hidden}
.roll-strip{display:flex;flex-direction:column;transition:transform .85s cubic-bezier(.22,.8,.18,1)}
.roll-d{height:1em;display:flex;align-items:center;justify-content:center}
#spark{width:100%;height:56px;margin-top:14px}
.spark-cap{margin-top:2px;font-size:.7rem;font-weight:500;color:var(--muted);letter-spacing:.02em}
.busyval{font-family:var(--display);font-weight:500;font-size:.8rem;color:var(--muted);
  font-variant-numeric:tabular-nums}
.tile.compact{display:flex;flex-direction:column}
.compact-mid{flex:1;display:flex;flex-direction:column;justify-content:center}

/* avg-time cards */
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:16px 0}
.card{text-align:left;cursor:pointer;background:var(--paper);border:1px solid var(--hair);
  border-radius:16px;box-shadow:var(--shadow);padding:18px 20px;color:var(--ink);
  font-family:inherit;transition:border-color .18s,transform .18s,box-shadow .18s}
.card:hover{border-color:#39477a;transform:translateY(-2px);box-shadow:var(--shadow-lg)}
.card .v{margin-top:9px;font-family:var(--display);font-weight:600;font-size:1.95rem;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums;color:var(--ink)}

/* checkout speed strip */
.speed{margin:24px 0 30px}
.speed-head{display:flex;align-items:baseline;gap:12px;margin:0 2px 12px}
.speed-head h2{margin:0;font-family:var(--display);font-weight:600;font-size:1.05rem;letter-spacing:-.01em}
.speed-head .note{font-size:.8rem;font-weight:500;color:var(--muted)}
.vs{display:grid;grid-template-columns:1fr auto 1fr;gap:0;align-items:stretch;
  background:var(--paper);border:1px solid var(--hair);border-radius:18px;
  box-shadow:var(--shadow);overflow:hidden}
.vs-side{padding:22px 26px}
.vs-side .who{font-weight:600;font-size:.82rem;letter-spacing:.01em}
.vs-side .v{margin-top:7px;font-family:var(--display);font-weight:600;font-size:2.3rem;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.vs-side .sub{margin-top:1px;font-size:.74rem;font-weight:500;color:var(--muted)}
.vs-side.mash{background:rgba(88,120,255,.10)}
.vs-side.mash .who{color:var(--brand-press)} .vs-side.mash .v{color:var(--brand)}
.vs-side.staff{text-align:right}
.vs-side.staff .who{color:var(--slate)} .vs-side.staff .v{color:#aab2c4}
.vs-mid{display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:0 22px;border-left:1px solid var(--hair);border-right:1px solid var(--hair);background:var(--sunken)}
.vs-mid .mult{font-family:var(--display);font-weight:700;font-size:2rem;letter-spacing:-.02em;
  color:var(--brand);font-variant-numeric:tabular-nums}
.vs-mid .sub{margin-top:2px;font-size:.66rem;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}

/* tabs + charts */
.tabs{display:inline-flex;gap:4px;padding:4px;background:var(--sunken);border:1px solid var(--hair);
  border-radius:13px;margin-bottom:16px}
.tabs button{border:0;background:transparent;color:var(--muted);font-family:var(--ui);font-weight:600;
  font-size:.86rem;padding:9px 18px;border-radius:9px;cursor:pointer;transition:.16s}
.tabs button.active{background:var(--paper);color:var(--brand-press);box-shadow:var(--shadow)}
.tabs button:hover:not(.active){color:var(--ink)}
.panel{background:var(--paper);border:1px solid var(--hair);border-radius:18px;
  box-shadow:var(--shadow);padding:14px 14px 8px}
.chart{width:100%;height:430px}
[hidden]{display:none!important}

/* debug tab */
.dbg{padding:6px 8px 10px}
.dbg-head{display:flex;align-items:baseline;gap:10px;margin:6px 4px 14px}
.dbg-frame{font-family:var(--display);font-weight:600;font-size:1.05rem;color:var(--ink);font-variant-numeric:tabular-nums}
.dbg-sub{color:var(--muted);font-size:.78rem;font-weight:500}
table.dbg-t{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
table.dbg-t th{text-align:left;font-size:.66rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);padding:9px 11px;border-bottom:1px solid var(--hair)}
table.dbg-t td{padding:10px 11px;border-bottom:1px solid var(--hair2);font-size:.9rem;font-weight:500}
table.dbg-t tr:hover td{background:#161e2e}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.72rem;font-weight:600}
.pill.waiting{color:#aab2c4;background:#1c2434}
.pill.mash{color:var(--brand-press);background:rgba(88,120,255,.16)}
.pill.reg{color:#aab2c4;background:var(--slate-wash)}
.pill.transit{color:#fbbf24;background:rgba(251,191,36,.13)}
.pill.cand{color:var(--brand-press);background:rgba(88,120,255,.13)}
.pill.out{color:var(--muted);background:var(--hair2)}
.flag{font-weight:700}.flag.on{color:var(--brand)}.flag.off{color:#3a4358}
.dbg-empty{padding:34px 12px;text-align:center;color:var(--muted);font-weight:500;line-height:1.7}
.foot{margin-top:18px;color:var(--muted);font-size:.76rem;font-weight:500;letter-spacing:.01em}

@media(max-width:820px){
  .hero{grid-template-columns:1fr 1fr}
  .tile.primary{grid-column:1 / -1}
  .cards{grid-template-columns:1fr}
  .vs{grid-template-columns:1fr;gap:0}
  .vs-side.staff{text-align:left}
  .vs-mid{border:0;border-top:1px solid var(--hair);border-bottom:1px solid var(--hair);padding:16px}
}
@media(prefers-reduced-motion:reduce){
  .roll-strip{transition:none}.dot{animation:none}
}
</style></head>
<body><div class="wrap">
<header>
  <div class="mark">
    <span class="logo"><b>Mashgin</b></span>
    <span class="div"></span>
    <span class="sub">Line Monitor</span>
  </div>
  <div class="live"><span class="lbl">Live</span><span class="dot"></span><span id="clock">connecting</span></div>
</header>

<section class="hero">
  <div class="tile primary" id="heroLine">
    <div class="tile-top">
      <span class="eyebrow">Currently in line</span>
      <span id="busy" class="chip">—</span>
    </div>
    <div class="rowend"><div class="bignum roller" id="rIn"></div><span class="unit">people</span></div>
    <div id="spark"></div>
    <div class="spark-cap">Live occupancy · last 90s · <span id="busyVal">busy index —</span></div>
  </div>
  <div class="tile compact">
    <span class="eyebrow">At checkout</span>
    <div class="compact-mid"><div class="rowend"><div class="bignum roller" id="rPos"></div><span class="unit">people</span></div></div>
  </div>
  <div class="tile compact">
    <span class="eyebrow">Served</span>
    <div class="compact-mid"><div class="rowend"><div class="bignum" id="served">0</div><span class="unit">visits</span></div></div>
  </div>
</section>

<section class="cards">
  <button class="card" data-tab="ws"><span class="eyebrow">Avg time in line</span><div class="v" id="avgLine">—</div></button>
  <button class="card" data-tab="ws"><span class="eyebrow">Avg at checkout</span><div class="v" id="avgPos">—</div></button>
  <button class="card" data-tab="ws"><span class="eyebrow">Avg total visit</span><div class="v" id="avgTotal">—</div></button>
</section>

<section class="speed" id="vs" hidden>
  <div class="speed-head">
    <h2>Checkout speed</h2>
    <span class="note">Mashgin self-checkout vs the staffed lane, measured live</span>
  </div>
  <div class="vs">
    <div class="vs-side mash"><div class="who">Mashgin self-checkout</div>
      <div class="v" id="vsM">—</div><div class="sub" id="vsMSub">avg time at the register</div></div>
    <div class="vs-mid"><div class="mult" id="vsMult">—</div><div class="sub">faster</div>
      <div class="sub" id="vsDelta" style="text-transform:none;letter-spacing:0;font-weight:500;margin-top:5px"></div></div>
    <div class="vs-side staff"><div class="who">Staffed lane</div>
      <div class="v" id="vsO">—</div><div class="sub">avg time at the register</div></div>
  </div>
</section>

<nav class="tabs">
  <button data-tab="occ" class="active">Occupancy</button>
  <button data-tab="ws">Wait &amp; checkout</button>
  <button data-tab="tp">Throughput</button>
  <button data-tab="cx">Lane comparison</button>
  <button data-tab="dbg">Debug</button>
</nav>
<section>
  <div class="panel" data-panel="occ"><div class="chart" id="chart-occ"></div></div>
  <div class="panel" data-panel="ws" hidden><div class="chart" id="chart-ws"></div></div>
  <div class="panel" data-panel="tp" hidden><div class="chart" id="chart-tp"></div></div>
  <div class="panel" data-panel="cx" hidden><div class="chart" id="chart-cx"></div></div>
  <div class="panel" data-panel="dbg" hidden>
    <div class="dbg">
      <div class="dbg-head"><span class="dbg-frame" id="dbgFrame">frame —</span>
        <span class="dbg-sub">per-person classification for the viewed frame</span></div>
      <div id="dbgBody"></div>
    </div>
  </div>
</section>
<div class="foot" id="foot">Waiting for the first frame…</div>
</div>
<script>
const C={ink:"#eaeef7",brand:"#5878ff",brandPress:"#8aa0ff",brandSoft:"#34407a",
  inLine:"#aeb8d0",inLineSoft:"#566182",slate:"#7c8699",mut:"#8b94a9",
  hair:"#232c3f",grid:"#1b2433",surface:"#141b2b"};
const pad=n=>String(n).padStart(2,"0");
const clk=s=>{s=Math.max(0,Math.round(s));return pad(Math.floor(s/60))+":"+pad(s%60)};
const dur=s=>{if(s==null||!isFinite(s))return "—";if(s<60)return s.toFixed(1)+"s";return Math.floor(s/60)+"m "+pad(Math.round(s%60))+"s"};

class Roller{constructor(el){this.el=el;this.n=null;}
  set(n){n=Math.max(0,Math.round(n||0));if(n===this.n)return;
    const ds=String(n).split("");
    if(this.el.children.length!==ds.length){this.el.innerHTML="";
      for(let k=0;k<ds.length;k++){const c=document.createElement("div");c.className="roll-col";
        const s=document.createElement("div");s.className="roll-strip";
        for(let i=0;i<10;i++){const d=document.createElement("div");d.className="roll-d";d.textContent=i;s.appendChild(d);}
        c.appendChild(s);this.el.appendChild(c);}}
    ds.forEach((d,i)=>{this.el.children[i].firstChild.style.transform=`translateY(${-d*10}%)`;});
    this.n=n;}}
const rIn=new Roller(document.getElementById("rIn"));
const rPos=new Roller(document.getElementById("rPos"));
rIn.set(0);rPos.set(0);

const charts={};
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>activate(b.dataset.tab));
document.querySelectorAll(".card").forEach(b=>b.onclick=()=>activate(b.dataset.tab));
function activate(tab){
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.toggle("active",b.dataset.tab===tab));
  document.querySelectorAll("[data-panel]").forEach(p=>p.hidden=(p.dataset.panel!==tab));
  if(charts[tab])charts[tab].resize();
  history.replaceState(null,"","#"+tab);
}
if(location.hash){const t=location.hash.slice(1);
  if(document.querySelector(`.tabs button[data-tab="${t}"]`))activate(t);}

function baseOpt(unit){return{
  backgroundColor:"transparent",
  textStyle:{fontFamily:"Inter, sans-serif",color:C.mut},
  grid:{left:46,right:26,top:38,bottom:62},
  legend:{top:6,right:6,textStyle:{color:C.ink,fontWeight:600},inactiveColor:"#49506880",icon:"roundRect",itemWidth:14,itemHeight:8},
  tooltip:{trigger:"axis",backgroundColor:C.surface,borderColor:C.hair,borderWidth:1,
    padding:[9,12],textStyle:{color:C.ink,fontFamily:"Inter, sans-serif"},
    extraCssText:"box-shadow:0 12px 34px rgba(0,0,0,.5);border-radius:12px",
    axisPointer:{type:"line",lineStyle:{color:"#39435c"}},
    formatter:p=>{let s=`<b>${clk(p[0].axisValue)}</b>`;p.forEach(x=>{s+=`<br>${x.marker}${x.seriesName}: <b>${(+x.value[1]).toFixed(unit==="s"?1:0)}${unit}</b>`;});return s;}},
  xAxis:{type:"value",min:"dataMin",
    axisLabel:{color:C.mut,formatter:clk,hideOverlap:true},axisLine:{lineStyle:{color:C.hair}},
    axisTick:{show:false},splitLine:{show:false}},
  yAxis:{type:"value",axisLabel:{color:C.mut},axisLine:{show:false},axisTick:{show:false},
    splitLine:{lineStyle:{color:C.grid}}},
  dataZoom:[{type:"inside"},{type:"slider",height:14,bottom:22,borderColor:C.hair,
    backgroundColor:C.surface,fillerColor:"rgba(88,120,255,.16)",
    handleStyle:{color:"#1a2233",borderColor:C.brand},moveHandleStyle:{color:C.brand},
    textStyle:{color:C.mut},dataBackground:{lineStyle:{color:"#2c3650"},areaStyle:{color:"#1a2233"}},
    selectedDataBackground:{lineStyle:{color:C.brand},areaStyle:{color:"rgba(88,120,255,.18)"}}}],
};}
function lineSeries(name,color,{dashed=false,fill=false,end=true,step=false}={}){return{
  name,type:"line",smooth:!step,step:step?"end":false,showSymbol:false,color,itemStyle:{color},
  lineStyle:{width:dashed?1.4:3,color,type:dashed?"dashed":"solid"},
  areaStyle:fill?{color:color+"22",opacity:1}:undefined,
  emphasis:{focus:"series"},
  endLabel:end?{show:true,color,fontWeight:700,fontFamily:"Space Grotesk, sans-serif",formatter:o=>` ${(+o.value[1]).toFixed(1)}`}:undefined,
  data:[]};}
function makeChart(id,build){const c=echarts.init(document.getElementById(id),null,{renderer:"canvas"});
  c.setOption(build());return c;}
// raw counts: faint dashed STEP lines; moving averages: solid bold lines (on top)
charts.occ=makeChart("chart-occ",()=>({...baseOpt(""),series:[
  lineSeries("in line (raw)",C.inLineSoft,{dashed:true,step:true,end:false}),
  lineSeries("at checkout (raw)",C.brandSoft,{dashed:true,step:true,end:false}),
  lineSeries("in line",C.inLine,{end:true}),
  lineSeries("at checkout",C.brand,{end:true})]}));
charts.ws=makeChart("chart-ws",()=>({...baseOpt("s"),series:[
  lineSeries("time in line (avg)",C.inLine,{fill:true}),lineSeries("at checkout (avg)",C.brand,{fill:true})]}));
charts.tp=makeChart("chart-tp",()=>({...baseOpt(""),legend:{show:false},series:[{
  name:"served/min",type:"bar",itemStyle:{color:C.brand,borderRadius:[4,4,0,0]},
  emphasis:{itemStyle:{color:C.brandPress}},data:[]}]}));
charts.cx=makeChart("chart-cx",()=>({...baseOpt("s"),series:[
  lineSeries("Mashgin",C.brand,{fill:true}),lineSeries("Staffed lane",C.slate,{fill:true})]}));

// hero sparkline — live in-line occupancy, no axes
charts.spark=echarts.init(document.getElementById("spark"),null,{renderer:"canvas"});
charts.spark.setOption({
  animation:false,grid:{left:0,right:0,top:6,bottom:2},
  xAxis:{type:"value",show:false,min:"dataMin",max:"dataMax"},
  yAxis:{type:"value",show:false,min:0},
  tooltip:{show:false},
  series:[{type:"line",smooth:true,showSymbol:false,color:C.brand,lineStyle:{width:2.5,color:C.brand},
    areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,
      [{offset:0,color:"rgba(88,120,255,.34)"},{offset:1,color:"rgba(88,120,255,0)"}])},data:[]}]});
window.addEventListener("resize",()=>{Object.values(charts).forEach(c=>c.resize());});

const zip=(t,v)=>t.map((x,i)=>[x,v[i]]);
const LVL={Low:"low",Medium:"med",High:"high"};
const WORD={Low:"Quiet",Medium:"Steady",High:"Busy"};

function statePill(r){
  if(r.transit)return '<span class="pill transit">transit</span>';
  if(r.state==="serving-Mashgin")return '<span class="pill mash">Mashgin checkout</span>';
  if(r.state==="serving-REG")return '<span class="pill reg">staffed lane</span>';
  if(r.state==="waiting")return '<span class="pill waiting">waiting</span>';
  if(r.state&&r.state.indexOf("candidate")===0)return `<span class="pill cand">${r.state}</span>`;
  return '<span class="pill out">out</span>';
}
const flag=on=>`<span class="flag ${on?"on":"off"}">${on?"●":"·"}</span>`;
function renderDebug(dbg){
  document.getElementById("dbgFrame").textContent=
    dbg && dbg.frame!=null ? `frame ${dbg.frame}` : "frame —";
  const body=document.getElementById("dbgBody");
  const rows=(dbg&&dbg.rows)||[];
  if(!rows.length){
    body.innerHTML='<div class="dbg-empty">No one in frame right now.<br>'+
      'Run with <b>--debug</b> to step frame-by-frame; this tab follows the viewed frame.</div>';
    return;}
  let h='<table class="dbg-t"><thead><tr>'+
    '<th>ID</th><th>State</th><th>In line</th><th>Checkout</th><th>Speed</th>'+
    '<th>Line</th><th>POS</th><th>Lane</th></tr></thead><tbody>';
  for(const r of rows){
    h+=`<tr><td>#${r.id}</td><td>${statePill(r)}</td>`+
       `<td>${dur(r.wait)}</td><td>${dur(r.serve)}</td>`+
       `<td>${(+r.speed).toFixed(2)}</td>`+
       `<td>${flag(r.line)}</td><td>${flag(r.pos)}</td><td>${flag(r.reg)}</td></tr>`;}
  body.innerHTML=h+'</tbody></table>';
}
async function poll(){
  try{
    const d=await (await fetch("metrics")).json();
    const s=d.summary;
    rIn.set(s.in_line);rPos.set(s.at_pos);
    document.getElementById("served").textContent=s.served_count;
    document.getElementById("avgLine").textContent=dur(s.avg_line_s);
    document.getElementById("avgPos").textContent=dur(s.avg_pos_s);
    document.getElementById("avgTotal").textContent=dur(s.avg_total_s);
    document.getElementById("clock").textContent=clk(d.now)+" elapsed";
    document.getElementById("foot").textContent=
      `Live from store camera · ${s.served_count} served`;

    // busy status drives the hero chip + left rail + caption
    const bz=d.busy&&d.busy.current, chip=document.getElementById("busy"),
      hero=document.getElementById("heroLine"), bv=document.getElementById("busyVal");
    if(bz&&bz.level){
      const cls=LVL[bz.level]||"";
      chip.className="chip "+cls;chip.textContent=WORD[bz.level]||bz.level;
      hero.className="tile primary "+cls;
      bv.textContent=`busy index ${(+bz.value).toFixed(1)}`;
    }else{chip.className="chip";chip.textContent="warming up";
      hero.className="tile primary";bv.textContent="busy index —";}

    const o=d.occupancy;
    charts.occ.setOption({series:[
      {data:zip(o.t,o.waiting)},{data:zip(o.t,o.serving)},
      {data:zip(o.t,o.waiting_ma)},{data:zip(o.t,o.serving_ma)}]});
    // sparkline: last 90 samples of raw in-line occupancy
    const N=90,ti=o.t.slice(-N),wi=o.waiting.slice(-N);
    charts.spark.setOption({series:[{data:zip(ti,wi)}]});

    const w=d.wait_serve;
    charts.ws.setOption({series:[{data:zip(w.t,w.wait_ma)},{data:zip(w.t,w.serve_ma)}]});
    const p=d.throughput;
    charts.tp.setOption({series:[{data:zip(p.t,p.served_per_min)}]});

    const ck=d.checkouts, vs=document.getElementById("vs");
    if(ck && ck.other_n>0){vs.hidden=false;
      // effective per-customer Mashgin time across N parallel kiosks
      const mEff=(ck.mashgin_avg_eff!=null)?ck.mashgin_avg_eff:ck.mashgin_avg;
      document.getElementById("vsM").textContent=dur(mEff);
      document.getElementById("vsO").textContent=dur(ck.other_avg);
      const n=ck.num_mashgins||1;
      document.getElementById("vsMSub").textContent=
        n>1?`across ${n} kiosks (${dur(ck.mashgin_avg)}/person)`:"avg time at the register";
      const mult=(mEff>0)?(ck.other_avg/mEff):null;
      document.getElementById("vsMult").textContent=mult?mult.toFixed(1)+"×":"—";
      document.getElementById("vsDelta").textContent=
        ck.delta>0?`saves ${dur(ck.delta)} / person`:"";}
    else{vs.hidden=true;}
    const cs=ck?ck.series:null;
    if(cs){charts.cx.setOption({series:[
      {data:zip(cs.t_mashgin,cs.mashgin_ma)},{data:zip(cs.t_other,cs.other_ma)}]});}
    renderDebug(d.debug);
  }catch(e){}
}
poll();setInterval(poll,1000);
</script></body></html>
"""
