import { defineConfig } from 'vite'
import { resolve } from 'node:path'

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5273,
    strictPort: true
  },
  preview: {
    host: '127.0.0.1',
    port: 5273,
    strictPort: true
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
