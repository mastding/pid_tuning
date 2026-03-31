export const findHelpTocParent = (toc, targetItem) => {
  const idx = toc.findIndex(item => item.id === targetItem.id);
  if (idx <= 0) return null;
  for (let i = idx - 1; i >= 0; i -= 1) {
    if (toc[i].level < targetItem.level) {
      return toc[i];
    }
  }
  return null;
};

export const isHelpTocExpanded = (expandedSections, sectionId) => {
  return Boolean(expandedSections?.[sectionId]);
};

export const isHelpTocVisible = (toc, expandedSections, item) => {
  if (item.level === 1) return true;
  const parent = findHelpTocParent(toc, item);
  if (!parent) return true;
  return isHelpTocExpanded(expandedSections, parent.id);
};

export const pickActiveSectionId = (measurements, thresholdPx = 140) => {
  if (!Array.isArray(measurements) || !measurements.length) return '';
  const current =
    measurements
      .filter(item => Number.isFinite(item.top))
      .filter(item => item.top <= thresholdPx)
      .sort((a, b) => b.top - a.top)[0]
    || measurements[0];
  return current?.id || '';
};

