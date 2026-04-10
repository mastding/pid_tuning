const ensureAppTemplate = async () => {
  const host = document.getElementById('app');
  if (!host) throw new Error('#app not found');

  if (host.childElementCount > 0) return;

  const candidatePaths = [
    `./app-template.html?v=${Date.now()}`,
    `./public/app-template.html?v=${Date.now()}`
  ];

  let html = '';
  let lastError = '';
  for (const path of candidatePaths) {
    const response = await fetch(path, { cache: 'no-store' });
    if (response.ok) {
      html = await response.text();
      break;
    }
    lastError = `HTTP ${response.status}`;
  }

  if (!html) throw new Error(lastError || 'app-template.html not found');

  const doc = new DOMParser().parseFromString(html, 'text/html');
  const source = doc.getElementById('app');
  if (!source) throw new Error('template #app not found in app-template.html');

  host.className = source.className || host.className;
  host.innerHTML = source.innerHTML;
};

await ensureAppTemplate();
await import('./app.js');
