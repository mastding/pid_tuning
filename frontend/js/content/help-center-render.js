const DEFAULT_SHORTCUT_LABELS = {
  '系统定位': '系统定位',
  '多智能体协作流程': '协作流程',
  '当前评分机制': '评分机制',
  '经验机制': '经验机制',
  '当前后端代码结构': '代码结构',
  '主要接口': '接口说明',
};

export const buildHelpCenterRenderModel = (markdown, markedInstance) => {
  const marked = markedInstance || window.marked;
  const rawHtml = marked ? marked.parse(markdown) : String(markdown || '');

  const wrapper = document.createElement('div');
  wrapper.innerHTML = rawHtml;

  const toc = [];
  let headingIndex = 0;
  wrapper.querySelectorAll('h1, h2, h3').forEach(node => {
    headingIndex += 1;
    const id = `help-section-${headingIndex}`;
    node.id = id;
    toc.push({
      id,
      text: (node.textContent || '').trim(),
      level: Number(node.tagName.substring(1)),
    });
  });

  const expandedSections = {};
  const normalizedToc = toc.map((item, index) => {
    const next = toc[index + 1];
    const hasChildren = Boolean(next && next.level > item.level);
    if (item.level === 1) {
      expandedSections[item.id] = true;
    }
    return { ...item, hasChildren };
  });

  const shortcuts = normalizedToc
    .filter(item => item.level <= 2)
    .filter(item => DEFAULT_SHORTCUT_LABELS[item.text])
    .map(item => ({ id: item.id, label: DEFAULT_SHORTCUT_LABELS[item.text] }));

  return {
    html: wrapper.innerHTML,
    toc: normalizedToc,
    expandedSections,
    shortcuts,
    activeSectionId: normalizedToc[0]?.id || ''
  };
};

