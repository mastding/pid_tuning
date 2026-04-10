import { defineConfig } from 'vite'
import { resolve } from 'node:path'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

// 读取.env文件
const envPath = join(dirname(new URL(import.meta.url).pathname), '..', '.env')
let backendPort = '4443'
let frontendPort = '5873'

try {
  const envContent = readFileSync(envPath, 'utf-8')
  const backendMatch = envContent.match(/BACKEND_PORT=(\d+)/)
  const frontendMatch = envContent.match(/FRONTEND_PORT=(\d+)/)
  if (backendMatch) backendPort = backendMatch[1]
  if (frontendMatch) frontendPort = frontendMatch[1]
} catch (e) {
  console.warn('.env file not found, using defaults')
}

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: parseInt(frontendPort),
    strictPort: true
  },
  preview: {
    host: '127.0.0.1',
    port: parseInt(frontendPort),
    strictPort: true
  },
  define: {
    __BACKEND_PORT__: JSON.stringify(backendPort)
  },
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        tuning: resolve(__dirname, 'tuning.html'),
        loopAnalysis: resolve(__dirname, 'loop-analysis.html'),
        experience: resolve(__dirname, 'experience.html'),
        caseLibrary: resolve(__dirname, 'case-library.html')
      }
    }
  }
})
