const ensureAppTemplate = async () => {
  const host = document.getElementById('app');
  if (!host) throw new Error('#app not found');

  if (host.childElementCount > 0) return;

  const response = await fetch(`/index.html?v=${Date.now()}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const html = await response.text();

  const doc = new DOMParser().parseFromString(html, 'text/html');
  const source = doc.getElementById('app');
  if (!source) throw new Error('template #app not found in index.html');

  host.className = source.className || host.className;
  host.innerHTML = source.innerHTML;
};

await ensureAppTemplate();
await import('./app.js');

