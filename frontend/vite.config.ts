import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    allowedHosts: ['soilx-ws.wpi.edu'],
    proxy: {
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
      },
      '/download': {
        target: 'http://localhost:8000',
      },
    },
  },
})
