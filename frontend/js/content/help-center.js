export const loadHelpCenterMarkdown = async () => {
  const response = await fetch(`./help-center.md?v=${Date.now()}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  return response.text();
};

