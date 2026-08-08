import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Local proxy avoids hard-coding a Bento origin in the frontend.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react(), tailwindcss()],
    server: {
      // Fail on an occupied IPv4 loopback port, matching Bento's bind host.
      host: '127.0.0.1',
      port: 5173,
      strictPort: true,
      proxy: {
        // Bento mounts routes at root and binds the IPv4 loopback.
        '/api': {
          target: env.VITE_BENTO_URL || 'http://127.0.0.1:3000',
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
