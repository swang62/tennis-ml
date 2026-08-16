import { writeFile } from "node:fs/promises";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type Plugin } from "vite";

// Build-time UTC date (YYYY-MM-DD) for the sitemap lastmod, computed once when
// the config module loads so both routes share one deterministic value per
// build. toISOString() is always UTC, so the date never depends on the build
// machine's timezone.
const SITEMAP_LASTMOD = new Date().toISOString().slice(0, 10);

// Adds environment-dependent SEO tags through Vite's structured
// transformIndexHtml return. When SITE_URL is unset, no wrong absolute URL is
// emitted.
function seoPlugin(siteUrl: string): Plugin {
  const base = siteUrl ? `${siteUrl.replace(/\/+$/, "")}/` : "";
  return {
    name: "courtside-seo",
    transformIndexHtml() {
      const jsonLd: Record<string, string> = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        name: "Courtside",
        description:
          "Courtside - model-driven tennis predictions, head-to-heads and player statistics.",
        applicationCategory: "SportsApplication",
        operatingSystem: "Web",
      };
      if (base) jsonLd.url = base;

      const tags = [
        {
          tag: "script",
          attrs: { type: "application/ld+json" },
          children: JSON.stringify(jsonLd, null, 2),
          injectTo: "head",
        },
      ];
      if (base) {
        tags.unshift(
          {
            tag: "link",
            attrs: { rel: "canonical", href: base },
            injectTo: "head-prepend",
          },
          {
            tag: "meta",
            attrs: { property: "og:url", content: base },
            injectTo: "head",
          },
        );
      }
      return { tags };
    },
  };
}

// Injects the Umami analytics script into index.html only when VITE_SITE_ID
// is set at build time. transformIndexHtml tag injection is Vite's documented
// mechanism for env-conditional head entries; the id comes from loadEnv
// (web/.env in dev, Docker ENV from --build-arg in image builds).
function umamiPlugin(siteId: string): Plugin {
  return {
    name: "courtside-umami",
    transformIndexHtml() {
      if (!siteId) return;
      return [
        {
          tag: "script",
          attrs: {
            defer: true,
            src: "https://umami.stronglybrewed.dev/script.js",
            "data-website-id": siteId,
            "data-performance": "true",
          },
          injectTo: "head",
        },
      ];
    },
  };
}

// public/ files are copied verbatim, so rewrite sitemap.xml in dist at build
// close with the build-time UTC lastmod in both VITE_SITE_URL modes (absolute
// locs plus a robots Sitemap line when set; relative locs and the verbatim
// public robots.txt when unset).
function seoFilesPlugin(siteUrl: string): Plugin {
  let outDir = "";
  return {
    name: "courtside-seo-files",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    async closeBundle() {
      if (!outDir) return;
      const base = siteUrl ? `${siteUrl.replace(/\/+$/, "")}/` : "";
      if (base) {
        const robots = `User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: ${base}sitemap.xml\n`;
        await writeFile(`${outDir}/robots.txt`, robots);
      }
      const locRoot = base || "/";
      const sitemap = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
        "  <url>",
        `    <loc>${locRoot}</loc>`,
        `    <lastmod>${SITEMAP_LASTMOD}</lastmod>`,
        "    <changefreq>weekly</changefreq>",
        "    <priority>1.0</priority>",
        "  </url>",
        "  <url>",
        `    <loc>${locRoot}h2h</loc>`,
        `    <lastmod>${SITEMAP_LASTMOD}</lastmod>`,
        "    <changefreq>weekly</changefreq>",
        "    <priority>0.8</priority>",
        "  </url>",
        "</urlset>",
        "",
      ].join("\n");
      await writeFile(`${outDir}/sitemap.xml`, sitemap);
    },
  };
}

// Local proxy avoids hard-coding a Bento origin in the frontend.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [
      react(),
      tailwindcss(),
      seoPlugin(env.VITE_SITE_URL || ""),
      seoFilesPlugin(env.VITE_SITE_URL || ""),
      umamiPlugin(env.VITE_SITE_ID || ""),
    ],
    server: {
      // Bind all interfaces so HMR is reachable from other devices on the LAN.
      host: "0.0.0.0",
      port: 5173,
      strictPort: true,
      proxy: {
        // Bento mounts routes at root and binds the IPv4 loopback.
        "/api": {
          target: env.VITE_BENTO_URL || "http://127.0.0.1:3000",
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
  };
});
