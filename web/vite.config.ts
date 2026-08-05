import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Local dev only. The `/api` proxy points at the locally-running Bento
// service (just deploy-local, :3000) so the browser never talks to it
// directly. No Dockerfile / deploy step — see the locked plan decision.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Strip the /api prefix: Bento mounts its routes at the root
      // (/players, /player_profile, /predict_from_ids, ...), not under /api.
      '/api': {
        target: 'http://localhost:3000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
