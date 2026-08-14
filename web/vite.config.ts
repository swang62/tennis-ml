import { writeFile } from 'node:fs/promises'
import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Fixed lastmod for the two static SPA routes so the emitted sitemap is deterministic.
const SITEMAP_LASTMOD = '2026-08-13'

// Replaces SEO markers in index.html with absolute URLs when SITE_URL is set at
// build time. When unset, the relative OG/Twitter tags stay and
// canonical/og:url/JSON-LD url are omitted so no wrong URL is emitted.
function seoPlugin(siteUrl: string): Plugin {
  const base = siteUrl ? `${siteUrl.replace(/\/+$/, '')}/` : ''
  return {
    name: 'courtside-seo',
    transformIndexHtml(html) {
      let out = html
        .replace('<!-- seo:canonical -->', base ? `<link rel="canonical" href="${base}" />` : '')
        .replace('<!-- seo:og-url -->', base ? `<meta property="og:url" content="${base}" />` : '')
      out = base
        ? out.replace('"@SITE_URL@"', `"${base}"`)
        : out.replace(',\n        "url": "@SITE_URL@"', '')
      return out
    },
  }
}

// public/ files are copied verbatim, so rewrite robots.txt and sitemap.xml in
// dist with absolute URLs when SITE_URL is set. When unset, the public/ copies
// (relative locs, no Sitemap line) are served as-is.
function seoFilesPlugin(siteUrl: string): Plugin {
  let outDir = ''
  return {
    name: 'courtside-seo-files',
    configResolved(config) {
      outDir = config.build.outDir
    },
    async closeBundle() {
      if (!siteUrl || !outDir) return
      const base = `${siteUrl.replace(/\/+$/, '')}/`
      const robots = `User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: ${base}sitemap.xml\n`
      const sitemap = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
        '  <url>',
        `    <loc>${base}</loc>`,
        `    <lastmod>${SITEMAP_LASTMOD}</lastmod>`,
        '    <changefreq>weekly</changefreq>',
        '    <priority>1.0</priority>',
        '  </url>',
        '  <url>',
        `    <loc>${base}h2h</loc>`,
        `    <lastmod>${SITEMAP_LASTMOD}</lastmod>`,
        '    <changefreq>weekly</changefreq>',
        '    <priority>0.8</priority>',
        '  </url>',
        '</urlset>',
        '',
      ].join('\n')
      await Promise.all([
        writeFile(`${outDir}/robots.txt`, robots),
        writeFile(`${outDir}/sitemap.xml`, sitemap),
      ])
    },
  }
}

// Local proxy avoids hard-coding a Bento origin in the frontend.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react(), tailwindcss(), seoPlugin(env.SITE_URL || ''), seoFilesPlugin(env.SITE_URL || '')],
    server: {
      // Bind all interfaces so HMR is reachable from other devices on the LAN.
      host: '0.0.0.0',
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
