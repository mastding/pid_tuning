export const createRafScheduler = () => {
  let handle = 0;
  return (fn) => {
    if (handle) cancelAnimationFrame(handle);
    handle = requestAnimationFrame(() => {
      handle = 0;
      fn();
    });
  };
};

export const renderPidAnalysisChart = (element, payload) => {
  if (!element || !window.PidAnalysisChart) return;
  if (!payload) {
    window.PidAnalysisChart.destroy(element);
    return;
  }
  window.PidAnalysisChart.render(element, payload);
};

export const destroyPidAnalysisChart = (element) => {
  if (!element || !window.PidAnalysisChart) return;
  window.PidAnalysisChart.destroy(element);
};

