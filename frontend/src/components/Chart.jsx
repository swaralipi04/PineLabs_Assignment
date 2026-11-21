import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

export default function Chart({ figure }) {
  if (!figure) return null;
  const fg = "#12332b";
  const grid = "rgba(46,139,94,0.15)";
  const base = figure.layout || {};
  const layout = {
    ...base,
    autosize: true,
    height: 360,
    width: undefined,
    font: { ...(base.font || {}), color: fg, size: 11 },
    title: { ...(typeof base.title === "object" ? base.title : { text: base.title }), font: { size: 13, color: fg }, x: 0.01 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    legend: { ...(base.legend || {}), orientation: "h", y: -0.22, x: 0, font: { size: 10.5, color: fg } },
    margin: { l: 56, r: 50, t: 44, b: 70 },
    xaxis: { ...(base.xaxis || {}), gridcolor: grid, color: fg },
    yaxis: { ...(base.yaxis || {}), gridcolor: grid, color: fg },
    ...(base.yaxis2 ? { yaxis2: { ...base.yaxis2, color: fg } } : {}),
  };
  return (
    <Plot
      data={figure.data}
      layout={layout}
      config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
      useResizeHandler
      style={{ width: "100%", height: 360 }}
    />
  );
}
