"""Self-contained dashboard HTML page: tabbed, interactive ECharts + odometer counters."""

PAGE_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>storePose · live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Azeret+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root{
  --bg:#0a0b0f; --panel:rgba(255,255,255,.035); --line:#ffb020; --pos:#37d6f5;
  --tp:#b692ff; --ink:#eef0f6; --muted:#888f9e; --hair:rgba(255,255,255,.08);
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Bricolage Grotesque",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
body{min-height:100vh;position:relative;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;z-index:-2;
  background:
   radial-gradient(60vw 60vw at 8% -10%, rgba(255,176,32,.10), transparent 60%),
   radial-gradient(60vw 60vw at 100% 110%, rgba(55,214,245,.10), transparent 60%),
   var(--bg)}
body::after{content:"";position:fixed;inset:0;z-index:-1;opacity:.4;pointer-events:none;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/><feColorMatrix type='saturate' values='0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)' opacity='.5'/></svg>")}
.wrap{max-width:1120px;margin:0 auto;padding:34px 26px 70px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:30px}
.brand{font-weight:800;font-size:1.45rem;letter-spacing:-.02em}
.brand b{color:var(--line)} .brand i{color:var(--pos);font-style:normal}
.live{display:flex;align-items:center;gap:9px;color:var(--muted);
  font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;background:#46e0a0;box-shadow:0 0 0 0 rgba(70,224,160,.6);
  animation:pulse 1.8s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(70,224,160,.55)}70%{box-shadow:0 0 0 9px rgba(70,224,160,0)}100%{box-shadow:0 0 0 0 rgba(70,224,160,0)}}

.counters{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
.counter{position:relative;padding:24px 26px;border:1px solid var(--hair);border-radius:20px;
  background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.012));overflow:hidden}
.counter .cap{display:flex;align-items:center;gap:9px;font-size:.74rem;letter-spacing:.2em;
  text-transform:uppercase;font-weight:600;color:var(--muted)}
.counter .cap::before{content:"";width:8px;height:8px;border-radius:2px}
.counter.line .cap::before{background:var(--line);box-shadow:0 0 12px var(--line)}
.counter.pos  .cap::before{background:var(--pos);box-shadow:0 0 12px var(--pos)}
.roller{display:flex;align-items:flex-end;gap:2px;margin-top:6px;
  font-family:"Azeret Mono",monospace;font-weight:700;line-height:1;
  font-size:clamp(3.2rem,11vw,6.6rem);letter-spacing:-.02em}
.counter.line .roller{color:var(--line);text-shadow:0 0 38px rgba(255,176,32,.35)}
.counter.pos  .roller{color:var(--pos);text-shadow:0 0 38px rgba(55,214,245,.35)}
.roll-col{height:1em;overflow:hidden}
.roll-strip{display:flex;flex-direction:column;transition:transform .9s cubic-bezier(.2,.85,.16,1)}
.roll-d{height:1em;display:flex;align-items:center;justify-content:center}
.counter .unit{margin-left:10px;font-family:"Bricolage Grotesque";font-weight:600;
  font-size:.95rem;color:var(--muted);align-self:center}

.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:34px}
.card{text-align:left;cursor:pointer;border:1px solid var(--hair);border-radius:16px;
  padding:18px 20px;background:var(--panel);color:var(--ink);transition:.25s;font-family:inherit}
.card:hover{border-color:rgba(255,255,255,.22);transform:translateY(-2px);
  background:rgba(255,255,255,.06)}
.card .k{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:600}
.card .v{margin-top:8px;font-family:"Azeret Mono",monospace;font-weight:700;font-size:2rem;letter-spacing:-.02em}
.card.line .v{color:var(--line)} .card.pos .v{color:var(--pos)} .card.tot .v{color:var(--ink)}

.tabs{display:inline-flex;gap:4px;padding:5px;border:1px solid var(--hair);border-radius:14px;
  background:var(--panel);margin-bottom:18px}
.tabs button{border:0;background:transparent;color:var(--muted);font-family:inherit;font-weight:600;
  font-size:.9rem;padding:9px 18px;border-radius:10px;cursor:pointer;transition:.2s;letter-spacing:.01em}
.tabs button.active{background:rgba(255,255,255,.10);color:var(--ink)}
.tabs button:hover:not(.active){color:var(--ink)}

.panel{border:1px solid var(--hair);border-radius:20px;background:var(--panel);padding:10px 8px 8px}
.chart{width:100%;height:430px}
[hidden]{display:none!important}
.foot{margin-top:18px;color:var(--muted);font-size:.74rem;letter-spacing:.04em}
</style></head>
<body><div class="wrap">
<header>
  <div class="brand"><b>store</b><i>Pose</i> · live</div>
  <div class="live"><span class="dot"></span><span id="clock">connecting</span></div>
</header>

<section class="counters">
  <div class="counter line"><div class="cap">In line</div>
    <div style="display:flex"><div class="roller" id="rIn"></div><span class="unit">people</span></div></div>
  <div class="counter pos"><div class="cap">At POS</div>
    <div style="display:flex"><div class="roller" id="rPos"></div><span class="unit">people</span></div></div>
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
const C={line:"#ffb020",pos:"#37d6f5",tp:"#b692ff",ma1:"#ffd98a",ma2:"#a8ecfb",ink:"#eef0f6",mut:"#888f9e",hair:"rgba(255,255,255,.10)"};
const pad=n=>String(n).padStart(2,"0");
const clk=s=>{s=Math.max(0,Math.round(s));return pad(Math.floor(s/60))+":"+pad(s%60)};
const dur=s=>{if(s==null||!isFinite(s))return "—";if(s<60)return s.toFixed(1)+"s";return Math.floor(s/60)+"m "+pad(Math.round(s%60))+"s"};

/* iOS-style odometer counter */
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

/* tabs */
const charts={};
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>activate(b.dataset.tab));
document.querySelectorAll(".card").forEach(b=>b.onclick=()=>activate(b.dataset.tab));
function activate(tab){
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.toggle("active",b.dataset.tab===tab));
  document.querySelectorAll("[data-panel]").forEach(p=>p.hidden=(p.dataset.panel!==tab));
  if(charts[tab]){charts[tab].resize();}
}

function baseOpt(unit){return{
  backgroundColor:"transparent",
  textStyle:{fontFamily:"Azeret Mono, monospace",color:C.mut},
  grid:{left:46,right:26,top:40,bottom:64},
  legend:{top:6,right:8,textStyle:{color:C.ink,fontFamily:"Bricolage Grotesque"},
          inactiveColor:"#555b68",icon:"roundRect"},
  tooltip:{trigger:"axis",backgroundColor:"rgba(16,18,24,.95)",borderColor:C.hair,
    textStyle:{color:C.ink},axisPointer:{type:"line",lineStyle:{color:"rgba(255,255,255,.25)"}},
    formatter:p=>{let s=`<b>${clk(p[0].axisValue)}</b>`;p.forEach(x=>{s+=`<br>${x.marker}${x.seriesName}: <b>${(+x.value[1]).toFixed(unit==="s"?1:0)}${unit}</b>`;});return s;}},
  xAxis:{type:"value",min:"dataMin",
    axisLabel:{color:C.mut,formatter:clk,hideOverlap:true},axisLine:{lineStyle:{color:C.hair}},
    splitLine:{show:false}},
  yAxis:{type:"value",axisLabel:{color:C.mut},splitLine:{lineStyle:{color:"rgba(255,255,255,.05)"}}},
  dataZoom:[{type:"inside"},{type:"slider",height:16,bottom:24,borderColor:C.hair,
    fillerColor:"rgba(255,255,255,.06)",handleStyle:{color:C.ink},
    textStyle:{color:C.mut},dataBackground:{lineStyle:{color:C.hair},areaStyle:{color:"rgba(255,255,255,.04)"}}}],
};}
function area(color){return new echarts.graphic.LinearGradient(0,0,0,1,
  [{offset:0,color:color+"55"},{offset:1,color:color+"02"}]);}
function lineSeries(name,color,{dashed=false,fill=false,end=true}={}){return{
  name,type:"line",smooth:true,showSymbol:false,color,itemStyle:{color},
  lineStyle:{width:dashed?1.5:2.4,color,type:dashed?"dashed":"solid"},
  areaStyle:fill?{color:area(color)}:undefined,
  emphasis:{focus:"series"},
  endLabel:end?{show:true,color,fontFamily:"Azeret Mono",fontWeight:700,
    formatter:o=>` ${(+o.value[1]).toFixed(1)}`}:undefined,
  data:[]};}

function makeChart(id,build){const c=echarts.init(document.getElementById(id),null,{renderer:"canvas"});
  c.setOption(build());return c;}
charts.occ=makeChart("chart-occ",()=>({...baseOpt(""),series:[
  lineSeries("in line",C.line,{fill:true}),lineSeries("at POS",C.pos,{fill:true}),
  lineSeries("in line avg",C.ma1,{dashed:true,end:false}),lineSeries("at POS avg",C.ma2,{dashed:true,end:false})]}));
charts.ws=makeChart("chart-ws",()=>({...baseOpt("s"),series:[
  lineSeries("wait (avg)",C.line,{fill:true}),lineSeries("serve (avg)",C.pos,{fill:true})]}));
charts.tp=makeChart("chart-tp",()=>({...baseOpt(""),legend:{show:false},series:[{
  name:"served/min",type:"bar",itemStyle:{color:area(C.tp),borderRadius:[4,4,0,0]},
  emphasis:{itemStyle:{color:C.tp}},data:[]}]}));
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
