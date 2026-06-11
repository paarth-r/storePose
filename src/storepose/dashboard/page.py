"""Self-contained dashboard HTML page — Mashgin-themed: light, clean, navy + teal."""

PAGE_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>storePose · live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root{
  --bg:#f6f8fb; --surface:#ffffff; --ink:#13243a; --navy:#16324f; --teal:#00bcd9;
  --muted:#67788c; --hair:#e7ecf2; --hair2:#eef2f6;
  --shadow:0 1px 2px rgba(16,38,58,.04),0 4px 16px rgba(16,38,58,.05);
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Hanken Grotesk",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1140px;margin:0 auto;padding:30px 26px 72px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:26px}
.brand{font-weight:800;font-size:1.4rem;letter-spacing:-.02em;color:var(--navy)}
.brand i{color:var(--teal);font-style:normal}
.live{display:flex;align-items:center;gap:8px;color:var(--muted);font-weight:600;
  font-size:.74rem;letter-spacing:.14em;text-transform:uppercase}
.dot{width:8px;height:8px;border-radius:50%;background:var(--teal);animation:p 2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.35}}

.counters{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.counter{background:var(--surface);border:1px solid var(--hair);border-radius:16px;
  box-shadow:var(--shadow);padding:22px 24px}
.cap{display:flex;align-items:center;gap:9px;font-size:.74rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.cap::before{content:"";width:9px;height:9px;border-radius:3px}
.counter.line .cap::before{background:var(--navy)}
.counter.pos  .cap::before{background:var(--teal)}
.rowend{display:flex;align-items:flex-end;gap:11px;margin-top:6px}
.roller{display:flex;line-height:1;font-weight:800;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1;
  font-size:clamp(3rem,9vw,5.4rem)}
.counter.line .roller{color:var(--navy)} .counter.pos .roller{color:var(--teal)}
.roll-col{height:1em;overflow:hidden}
.roll-strip{display:flex;flex-direction:column;transition:transform .8s cubic-bezier(.3,.85,.2,1)}
.roll-d{height:1em;display:flex;align-items:center;justify-content:center}
.unit{font-weight:600;font-size:.95rem;color:var(--muted);padding-bottom:.5rem}

.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:30px}
.card{text-align:left;cursor:pointer;background:var(--surface);border:1px solid var(--hair);
  border-radius:14px;box-shadow:var(--shadow);padding:16px 19px;color:var(--ink);
  font-family:inherit;transition:border-color .18s,transform .18s}
.card:hover{border-color:#cfd9e4;transform:translateY(-1px)}
.card .k{font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.card .v{margin-top:6px;font-weight:800;font-size:1.85rem;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.card.line .v{color:var(--navy)} .card.pos .v{color:var(--teal)} .card.tot .v{color:var(--ink)}

.tabs{display:inline-flex;gap:3px;padding:4px;background:var(--hair2);border-radius:12px;margin-bottom:16px}
.tabs button{border:0;background:transparent;color:var(--muted);font-family:inherit;font-weight:700;
  font-size:.86rem;padding:8px 17px;border-radius:9px;cursor:pointer;transition:.16s}
.tabs button.active{background:var(--surface);color:var(--navy);box-shadow:var(--shadow)}
.tabs button:hover:not(.active){color:var(--navy)}

.panel{background:var(--surface);border:1px solid var(--hair);border-radius:16px;
  box-shadow:var(--shadow);padding:12px 12px 6px}
.chart{width:100%;height:430px}
[hidden]{display:none!important}
.foot{margin-top:16px;color:var(--muted);font-size:.76rem;font-weight:500}
</style></head>
<body><div class="wrap">
<header>
  <div class="brand">store<i>Pose</i> · live</div>
  <div class="live"><span class="dot"></span><span id="clock">connecting</span></div>
</header>

<section class="counters">
  <div class="counter line"><div class="cap">In line</div>
    <div class="rowend"><div class="roller" id="rIn"></div><span class="unit">people</span></div></div>
  <div class="counter pos"><div class="cap">At POS</div>
    <div class="rowend"><div class="roller" id="rPos"></div><span class="unit">people</span></div></div>
</section>

<section class="cards">
  <button class="card line" data-tab="ws"><div class="k">Avg line time</div><div class="v" id="avgLine">—</div></button>
  <button class="card pos"  data-tab="ws"><div class="k">Avg POS time</div><div class="v" id="avgPos">—</div></button>
  <button class="card tot"  data-tab="ws"><div class="k">Avg total time</div><div class="v" id="avgTotal">—</div></button>
</section>

<nav class="tabs">
  <button data-tab="occ" class="active">Occupancy</button>
  <button data-tab="ws">Wait &amp; Serve</button>
  <button data-tab="tp">Throughput</button>
</nav>
<section>
  <div class="panel" data-panel="occ"><div class="chart" id="chart-occ"></div></div>
  <div class="panel" data-panel="ws" hidden><div class="chart" id="chart-ws"></div></div>
  <div class="panel" data-panel="tp" hidden><div class="chart" id="chart-tp"></div></div>
</section>
<div class="foot" id="foot">served 0 · waiting for data…</div>
</div>
<script>
const C={navy:"#16324f",teal:"#00bcd9",navySoft:"#8aa0b6",tealSoft:"#7fd7e6",
  ink:"#13243a",mut:"#67788c",hair:"#e7ecf2",grid:"#eef2f6"};
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
}

function baseOpt(unit){return{
  backgroundColor:"transparent",
  textStyle:{fontFamily:"Hanken Grotesk, sans-serif",color:C.mut},
  grid:{left:44,right:24,top:38,bottom:62},
  legend:{top:6,right:6,textStyle:{color:C.ink,fontWeight:600},inactiveColor:"#b9c4d0",icon:"roundRect",itemWidth:14,itemHeight:8},
  tooltip:{trigger:"axis",backgroundColor:"#ffffff",borderColor:C.hair,borderWidth:1,
    padding:[8,11],textStyle:{color:C.ink},extraCssText:"box-shadow:0 6px 20px rgba(16,38,58,.12);border-radius:10px",
    axisPointer:{type:"line",lineStyle:{color:"#c6d0db"}},
    formatter:p=>{let s=`<b>${clk(p[0].axisValue)}</b>`;p.forEach(x=>{s+=`<br>${x.marker}${x.seriesName}: <b>${(+x.value[1]).toFixed(unit==="s"?1:0)}${unit}</b>`;});return s;}},
  xAxis:{type:"value",min:"dataMin",
    axisLabel:{color:C.mut,formatter:clk,hideOverlap:true},axisLine:{lineStyle:{color:C.hair}},
    axisTick:{show:false},splitLine:{show:false}},
  yAxis:{type:"value",axisLabel:{color:C.mut},axisLine:{show:false},axisTick:{show:false},
    splitLine:{lineStyle:{color:C.grid}}},
  dataZoom:[{type:"inside"},{type:"slider",height:14,bottom:22,borderColor:C.hair,
    backgroundColor:"#fbfcfe",fillerColor:"rgba(0,188,217,.10)",
    handleStyle:{color:"#fff",borderColor:C.navy},moveHandleStyle:{color:C.navy},
    textStyle:{color:C.mut},dataBackground:{lineStyle:{color:"#d7dee6"},areaStyle:{color:"#eef2f6"}},
    selectedDataBackground:{lineStyle:{color:C.teal},areaStyle:{color:"rgba(0,188,217,.12)"}}}],
};}
function lineSeries(name,color,{dashed=false,fill=false,end=true,step=false}={}){return{
  name,type:"line",smooth:!step,step:step?"end":false,showSymbol:false,color,itemStyle:{color},
  lineStyle:{width:dashed?1.4:3,color,type:dashed?"dashed":"solid"},
  areaStyle:fill?{color:color+"14",opacity:1}:undefined,
  emphasis:{focus:"series"},
  endLabel:end?{show:true,color,fontWeight:700,formatter:o=>` ${(+o.value[1]).toFixed(1)}`}:undefined,
  data:[]};}
function makeChart(id,build){const c=echarts.init(document.getElementById(id),null,{renderer:"canvas"});
  c.setOption(build());return c;}
// raw counts: faint dashed STEP lines; moving averages: solid bold lines (on top)
charts.occ=makeChart("chart-occ",()=>({...baseOpt(""),series:[
  lineSeries("in line (raw)",C.navySoft,{dashed:true,step:true,end:false}),
  lineSeries("at POS (raw)",C.tealSoft,{dashed:true,step:true,end:false}),
  lineSeries("in line",C.navy,{end:true}),
  lineSeries("at POS",C.teal,{end:true})]}));
charts.ws=makeChart("chart-ws",()=>({...baseOpt("s"),series:[
  lineSeries("wait (avg)",C.navy,{fill:true}),lineSeries("serve (avg)",C.teal,{fill:true})]}));
charts.tp=makeChart("chart-tp",()=>({...baseOpt(""),legend:{show:false},series:[{
  name:"served/min",type:"bar",itemStyle:{color:C.teal,borderRadius:[3,3,0,0]},
  emphasis:{itemStyle:{color:C.navy}},data:[]}]}));
window.addEventListener("resize",()=>Object.values(charts).forEach(c=>c.resize()));

const zip=(t,v)=>t.map((x,i)=>[x,v[i]]);
async function poll(){
  try{
    const d=await (await fetch("metrics")).json();
    const s=d.summary;
    rIn.set(s.in_line);rPos.set(s.at_pos);
    document.getElementById("avgLine").textContent=dur(s.avg_line_s);
    document.getElementById("avgPos").textContent=dur(s.avg_pos_s);
    document.getElementById("avgTotal").textContent=dur(s.avg_total_s);
    document.getElementById("clock").textContent=clk(d.now)+" elapsed";
    document.getElementById("foot").textContent=`served ${s.served_count} · live`;
    const o=d.occupancy;
    charts.occ.setOption({series:[
      {data:zip(o.t,o.waiting)},{data:zip(o.t,o.serving)},
      {data:zip(o.t,o.waiting_ma)},{data:zip(o.t,o.serving_ma)}]});
    const w=d.wait_serve;
    charts.ws.setOption({series:[{data:zip(w.t,w.wait_ma)},{data:zip(w.t,w.serve_ma)}]});
    const p=d.throughput;
    charts.tp.setOption({series:[{data:zip(p.t,p.served_per_min)}]});
  }catch(e){}
}
poll();setInterval(poll,1000);
</script></body></html>
"""
