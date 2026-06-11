"""Self-contained dashboard HTML page (Chart.js via CDN, polls /metrics)."""

PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>storePose dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background:#111; color:#eee; }
  h1 { font-size: 18px; } .chart { max-width: 900px; margin-bottom: 28px; }
</style></head><body>
<h1>storePose &mdash; live</h1>
<div class="chart"><canvas id="occ"></canvas></div>
<div class="chart"><canvas id="ws"></canvas></div>
<div class="chart"><canvas id="tp"></canvas></div>
<script>
function mk(id, title, series) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { datasets: series.map(s => ({label: s.label, data: [], borderColor: s.color,
            borderWidth: 2, pointRadius: 0, tension: 0.2})) },
    options: { animation: false, responsive: true,
      plugins: { title: { display: true, text: title, color: '#eee' },
                 legend: { labels: { color: '#eee' } } },
      scales: { x: { type: 'linear', title: { display: true, text: 'seconds', color:'#aaa' },
                     ticks:{color:'#aaa'} }, y: { beginAtZero: true, ticks:{color:'#aaa'} } } }
  });
}
const occ = mk('occ', 'Occupancy', [
  {label:'in line', color:'#ffb300'}, {label:'at POS', color:'#00c8ff'},
  {label:'in line (avg)', color:'#ffe082'}, {label:'at POS (avg)', color:'#80deea'}]);
const ws = mk('ws', 'Wait & serve (moving avg, s)', [
  {label:'wait', color:'#ff5252'}, {label:'serve', color:'#69f0ae'}]);
const tp = mk('tp', 'Throughput (served/min)', [{label:'served/min', color:'#b388ff'}]);
function xy(t, v){ return t.map((tt,i)=>({x:tt, y:v[i]})); }
async function poll(){
  try {
    const r = await fetch('metrics'); const d = await r.json();
    const o = d.occupancy;
    occ.data.datasets[0].data = xy(o.t, o.waiting);
    occ.data.datasets[1].data = xy(o.t, o.serving);
    occ.data.datasets[2].data = xy(o.t, o.waiting_ma);
    occ.data.datasets[3].data = xy(o.t, o.serving_ma);
    occ.update();
    const w = d.wait_serve;
    ws.data.datasets[0].data = xy(w.t, w.wait_ma);
    ws.data.datasets[1].data = xy(w.t, w.serve_ma);
    ws.update();
    const p = d.throughput;
    tp.data.datasets[0].data = xy(p.t, p.served_per_min);
    tp.update();
  } catch (e) {}
}
poll(); setInterval(poll, 1000);
</script></body></html>
"""
