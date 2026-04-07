import Plotly from 'plotly.js-dist-min';

const COLORS = {
  pv: '#1d4ed8',
  pvPred: '#7c3aed',
  sv: '#222222',
  mv: 'rgba(245, 158, 11, 0.78)',
  mvPred: 'rgba(217, 119, 6, 0.65)',
  fill: 'rgba(59, 130, 246, 0.12)',
  grid: '#e5e7eb',
  axis: '#475569'
};

const buildHoverText = (item, showFit) => {
  const time = item.label || '-';
  const pv = Number.isFinite(item.pv) ? item.pv.toFixed(3) : '-';
  const pvFit = Number.isFinite(item.pv_fit) ? item.pv_fit.toFixed(3) : null;
  const sv = Number.isFinite(item.sv) ? item.sv.toFixed(3) : '-';
  const svRaw = Number.isFinite(item.sv_raw) ? item.sv_raw.toFixed(3) : null;
  const mv = Number.isFinite(item.mv) ? item.mv.toFixed(3) : '-';
  const error = Number.isFinite(item.error) ? item.error.toFixed(3) : '-';
  const lines = [
    `时间：${time}`,
    `PV：${pv}`,
    ...(showFit && pvFit !== null ? [`PV(拟合)：${pvFit}`] : []),
    `SV：${sv}`,
    ...(svRaw !== null ? [`SV(原始)：${svRaw}`] : []),
    `MV：${mv}`,
    `误差(SV-PV)：${error}`
  ];
  return lines.join('<br>');
};

const buildPredHoverText = (item) => {
  const time = item.label || '-';
  const pvPred = Number.isFinite(item.pv_pred) ? item.pv_pred.toFixed(3) : '-';
  const mvPred = Number.isFinite(item.mv_pred) ? item.mv_pred.toFixed(3) : '-';
  const sv = Number.isFinite(item.sv) ? item.sv.toFixed(3) : '-';
  const errorPred = Number.isFinite(item.pv_pred) && Number.isFinite(item.sv) ? (item.sv - item.pv_pred).toFixed(3) : '-';
  return [
    `时间：${time}`,
    `PV(预测)：${pvPred}`,
    `SV：${sv}`,
    `MV(预测)：${mvPred}`,
    `误差(SV-PV预测)：${errorPred}`
  ].join('<br>');
};

const buildHighlightShapes = (payload) => {
  const ranges = Array.isArray(payload?.highlightRanges) ? payload.highlightRanges : [];
  const points = Array.isArray(payload?.points) ? payload.points : [];
  const resolveLabelByIndex = (targetIndex) => {
    const exact = points.find((point) => Number(point?.index) === Number(targetIndex));
    if (exact?.label) return exact.label;
    const numericTarget = Number(targetIndex);
    if (!Number.isFinite(numericTarget)) return null;
    let nearest = null;
    let minGap = Number.POSITIVE_INFINITY;
    points.forEach((point) => {
      const pointIndex = Number(point?.index);
      if (!Number.isFinite(pointIndex) || !point?.label) return;
      const gap = Math.abs(pointIndex - numericTarget);
      if (gap < minGap) {
        minGap = gap;
        nearest = point.label;
      }
    });
    return nearest;
  };

  return ranges
    .map((range) => {
      const hasIndexRange = Number.isFinite(Number(range?.startIndex)) && Number.isFinite(Number(range?.endIndex));
      const startLabel = range?.startLabel || (hasIndexRange ? resolveLabelByIndex(range.startIndex) : null);
      const endLabel = range?.endLabel || (hasIndexRange ? resolveLabelByIndex(range.endIndex) : null);
      const hasLabelRange = startLabel && endLabel;
      if (!hasIndexRange && !hasLabelRange) return null;
      return {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: hasLabelRange ? startLabel : Number(range.startIndex),
        x1: hasLabelRange ? endLabel : Number(range.endIndex),
        y0: 0,
        y1: 1,
        fillcolor: range?.color || 'rgba(59, 130, 246, 0.12)',
        line: { width: 0 },
        layer: 'below'
      };
    })
    .filter(Boolean);
};

export const render = (element, payload) => {
  if (!element || !payload || !Array.isArray(payload.points) || !payload.points.length) {
    return;
  }

  const showFit = Boolean(payload.showFit);
  const showSV = payload.showSV !== false;
  const showMV = payload.showMV !== false;

  const labels = payload.points.map(item => item.label);
  const pv = payload.points.map(item => item.pv);
  const pvFit = payload.points.map(item => item.pv_fit ?? item.pvFit);
  const pvPred = payload.points.map(item => item.pv_pred ?? item.pvPred);
  const sv = payload.points.map(item => item.sv);
  const mv = payload.points.map(item => item.mv);
  const mvPred = payload.points.map(item => item.mv_pred ?? item.mvPred);
  const hoverText = payload.points.map(item => buildHoverText(item, showFit));
  const predHoverText = payload.points.map(buildPredHoverText);

  const traces = [
    {
      x: labels,
      y: pv,
      name: 'PV',
      mode: 'lines',
      line: {
        color: COLORS.pv,
        width: 2.4
      },
      hovertemplate: '%{hovertext}<extra></extra>',
      hovertext: hoverText
    },
  ];

  if (showSV && sv.some(value => Number.isFinite(value))) {
    traces.push({
      x: labels,
      y: sv,
      name: 'SV',
      mode: 'lines',
      line: {
        color: COLORS.sv,
        width: 2,
        dash: 'dash'
      },
      fill: 'tonexty',
      fillcolor: COLORS.fill,
      hoverinfo: 'skip'
    });
  }

  if (showFit && pvFit.some(value => Number.isFinite(value))) {
    traces.push({
      x: labels,
      y: pvFit,
      name: 'PV(拟合)',
      mode: 'lines',
      line: {
        color: '#059669',
        width: 2,
        dash: 'dot'
      },
      hoverinfo: 'skip'
    });
  }

  if (pvPred.some(value => Number.isFinite(value))) {
    traces.push({
      x: labels,
      y: pvPred,
      name: 'PV(预测)',
      mode: 'lines',
      line: {
        color: COLORS.pvPred,
        width: 2,
        dash: 'dot'
      },
      hoverinfo: 'skip'
    });
  }

  if (showMV && mv.some(value => Number.isFinite(value))) {
    traces.push({
      x: labels,
      y: mv,
      name: 'MV',
      mode: 'lines',
      yaxis: 'y2',
      line: {
        color: COLORS.mv,
        width: 2.2
      },
      opacity: 0.95,
      hoverinfo: 'skip'
    });
  }

  if (showMV && mvPred.some(value => Number.isFinite(value))) {
    traces.push({
      x: labels,
      y: mvPred,
      name: 'MV(预测)',
      mode: 'lines',
      yaxis: 'y2',
      line: {
        color: COLORS.mvPred,
        width: 1.8,
        dash: 'dot'
      },
      opacity: 0.85,
      hoverinfo: 'skip'
    });
  }

  const layout = {
    margin: { l: 64, r: 72, t: 54, b: 78 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: {
      family: 'Microsoft YaHei, Segoe UI, sans-serif',
      size: 12,
      color: '#0f172a'
    },
    hovermode: 'x unified',
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.18,
      bgcolor: 'rgba(255,255,255,0.9)'
    },
    xaxis: {
      title: payload.xAxisTitle || '时间',
      type: payload.xAxisType === 'timestamp' ? 'date' : 'linear',
      gridcolor: COLORS.grid,
      linecolor: COLORS.grid,
      tickfont: { color: COLORS.axis },
      rangeslider: {
        visible: false
      }
    },
    yaxis: {
      title: payload.leftAxisTitle || 'PV / SV',
      gridcolor: COLORS.grid,
      zerolinecolor: COLORS.grid,
      linecolor: COLORS.grid,
      tickfont: { color: COLORS.axis }
    }
  };

  const highlightShapes = buildHighlightShapes(payload);

  if (showMV) {
    layout.yaxis2 = {
      title: payload.rightAxisTitle || 'MV (%)',
      overlaying: 'y',
      side: 'right',
      range: [0, 100],
      showgrid: false,
      tickfont: { color: COLORS.axis }
    };
    layout.shapes = [
      ...highlightShapes,
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y2',
        y0: 0,
        y1: 0,
        line: { color: '#f1f5f9', width: 1 }
      },
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y2',
        y0: 100,
        y1: 100,
        line: { color: '#f1f5f9', width: 1 }
      }
    ];
  } else if (highlightShapes.length) {
    layout.shapes = highlightShapes;
  }

  const config = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
  };

  try {
    Plotly.react(element, traces, layout, config);
  } catch (error) {
    console.warn('[PidAnalysisChart] Plotly.react failed, retry with newPlot:', error);
    Plotly.purge(element);
    Plotly.newPlot(element, traces, layout, config);
  }

  window.requestAnimationFrame(() => {
    try {
      Plotly.Plots?.resize?.(element);
    } catch (resizeError) {
      console.warn('[PidAnalysisChart] resize failed:', resizeError);
    }
  });
};

export const destroy = (element) => {
  if (element) Plotly.purge(element);
};

if (typeof window !== 'undefined') {
  window.PidAnalysisChart = { render, destroy };
}
