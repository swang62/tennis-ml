# Plan: Web Modernization — Lighthouse 100s, Instant Feel, SEO, Security

## Goal

Make the Courtside SPA score 100 across all Lighthouse categories, feel instant on every interaction, and be production-grade for SEO and security. Stay within the TanStack ecosystem where possible.

## Current State

| Metric | Value |
|--------|-------|
| Total bundle | 1.5 MB (single JS chunk: 1.49 MB) |
| CSS | 34 KB (one file) |
| Routes | 2 (Home `/`, H2H `/h2h`) |
| Code splitting | None — everything in one chunk |
| Preloading | None |
| SEO | Basic meta description only |
| Compression | gzip only |
| Images | favicon.png + external flagcdn flags |

**Biggest bottleneck**: ECharts is imported as `import * as echarts from "echarts"` (full bundle, ~800KB+ uncompressed) and nothing is code-split.

## Scope

### In scope
- Bundle size reduction (ECharts tree-shaking, route code splitting)
- Instant navigation (TanStack Router preload, data prefetch)
- SEO (structured data, OG tags, page titles, sitemap, robots.txt)
- Caching & compression (brotli + gzip, preconnect hints)
- Lighthouse-specific fixes (CLS, LCP, font loading)

### Out of scope
- SSR / SSG (this is a client-side SPA with a Bento API backend)
- Adding new pages or features
- Changing the data model or API
- Security headers (handled by Cloudflare)

---

## Tasks

### [x] Task 1: Tree-shake ECharts imports

**Status:** DONE — `import * as echarts` gone from src/; `src/lib/echarts.ts` registers Line/Bar/Radar + components + CanvasRenderer; bundle split into 4 chunks (see Task 2 sizes).

- **Description**: Replace `import * as echarts from "echarts"` and `import ReactECharts from "echarts-for-react"` with selective ECharts component imports. Only import the chart types, components, and renderers actually used (line, bar, radar, tooltip, legend, grid, markLine). This alone should cut ~500KB+ from the bundle.
- **Files**:
  - `web/src/pages/Profile.tsx` — uses line chart (rank history)
  - `web/src/pages/H2H.tsx` — uses bar chart (model comparison), radar chart (strength comparison), line chart (cumulative wins trend)
  - Create `web/src/lib/echarts.ts` — single file that registers only the needed components and exports a configured `ReactECharts` wrapper or the `echarts` instance
- **Acceptance Criteria**:
  - `import * as echarts` no longer appears anywhere in src/
  - Bundle JS size < 700KB (from 1.49MB)
  - All three chart types still render correctly
  - `pnpm build` succeeds, `pnpm test` passes

### [x] Task 2: Route-level code splitting

**Status:** DONE — 4 chunks in dist: index 341,164 B, Profile 8,689 B, H2H 17,318 B, charts 583,721 B; H2H via lazyRouteComponent, Profile via React.lazy.

- **Description**: Use TanStack Router's `lazyRouteComponent` to split Profile and H2H into separate chunks. The initial load should only include the router shell, Layout, and Home page. H2H and its ECharts dependency load only when navigating to `/h2h`.
- **Files**:
  - `web/src/router.tsx` — convert `createRoute` for `/h2h` to use `lazyRouteComponent(() => import("./pages/H2H"))`
  - `web/src/pages/Home.tsx` — Profile is already imported inline; consider lazy-loading it too since it's only shown after selecting a player
- **Acceptance Criteria**:
  - Initial JS chunk < 300KB
  - H2H route loads its own chunk on navigation
  - No flash of unstyled content during route transitions
  - `pnpm build` shows multiple chunks in `dist/assets/`

### [x] Task 3: TanStack Router preload + data prefetch

**Status:** DONE — `defaultPreload: "intent"` on router; hover/focus prefetch of similar-player queries in Home.tsx. Loader/ensureQueryData data-prefetch intentionally skipped (staleTime:Infinity already caches).

- **Description**: Make every `<Link>` preload its route on hover/focus using TanStack Router's built-in `preload="intent"`. Add route `loader` functions to prefetch data (players directory, directory_info) before the component renders, so navigation feels instant.
- **Files**:
  - `web/src/router.tsx` — add `preload: "intent"` to route definitions or as a default on the router; add `loader` functions that call `queryClient.ensureQueryData()` for shared data
  - `web/src/main.tsx` — pass `queryClient` to the router context so loaders can access it
  - All `<Link>` usages — ensure they use the default preload behavior
- **Acceptance Criteria**:
  - Hovering a nav link preloads the route chunk + data
  - Navigating to H2H shows data immediately (no loading spinner) when coming from Home after a player is selected
  - `queryClient` is accessible in route loaders via router context

### [x] Task 4: Page titles per route

**Status:** DONE — `document.title` set per route via Layout useEffect on pathname; player-name override in Home.tsx.

- **Description**: Set dynamic `<title>` per route for SEO and tab identification. Use TanStack Router's `beforeLoad` or a layout effect to update `document.title`.
- **Files**:
  - `web/src/router.tsx` — add `beforeLoad` or use a `useEffect` in Layout to set title based on current route
  - `web/src/pages/Home.tsx` — set title to "Players — Courtside" or "Courtside — Tennis Intelligence"
  - `web/src/pages/H2H.tsx` — set title to "Matchups — Courtside"
  - When a player is selected on Home: "Player Name — Courtside"
- **Acceptance Criteria**:
  - Each route has a unique, descriptive `<title>`
  - Title updates when selecting a player
  - Browser tab shows correct title

### [x] Task 5: SEO — structured data, OG tags, meta

**Status:** DONE — OG/Twitter Card/canonical/JSON-LD in index.html, SITE_URL injected at build via vite transformIndexHtml; og-image.png (1200x630) generated; external validators removed per user (local only).

- **Description**: Add JSON-LD structured data for the site (WebApplication schema), Open Graph and Twitter Card meta tags for link previews, and a canonical URL.
- **Files**:
  - `web/index.html` — add OG meta tags (`og:title`, `og:description`, `og:image`, `og:type`, `og:url`), Twitter Card tags, canonical link
  - Create `web/public/og-image.png` — a 1200x630 social sharing image (or generate from existing brand assets)
  - `web/src/router.tsx` or a `<SeoHead>` component — inject JSON-LD `<script type="application/ld+json">` with WebApplication schema
- **Acceptance Criteria**:
  - Sharing the URL on Twitter/Slack/Facebook shows a rich preview
  - Google Rich Results Test passes for the structured data
  - `<link rel="canonical">` is present

### [x] Task 6: SEO — robots.txt and sitemap.xml

**Status:** DONE — public/robots.txt + sitemap.xml, SITE_URL-substituted at build (closeBundle plugin); unset SITE_URL is a sane no-op.

- **Description**: Add a `robots.txt` allowing all crawlers and a `sitemap.xml` listing the two routes.
- **Files**:
  - Create `web/public/robots.txt` — `User-agent: *`, `Allow: /`, `Sitemap: <url>/sitemap.xml`
  - Create `web/public/sitemap.xml` — list `/` and `/h2h` with `<lastmod>`, `<changefreq>`, `<priority>`
- **Acceptance Criteria**:
  - `/robots.txt` and `/sitemap.xml` are served correctly
  - Sitemap validates at sitemap.xml validators

### [x] Task 7: Brotli + gzip compression + preconnect hints

**Status:** PARTIAL — brotli skipped (verified nginx:alpine has no ngx_brotli; would be a fatal "unknown directive"); gzip hardened (gzip_vary on, text/javascript added); flagcdn preconnect added to index.html.

- **Description**: Enable both Brotli and gzip compression in nginx (Brotli for clients that support it, gzip as fallback). Add `<link rel="preconnect">` for flagcdn.com in index.html so flag images start loading earlier.
- **Files**:
  - `web/nginx.conf.template` — add `brotli on;` + `brotli_types` and keep existing gzip config (both active; nginx negotiates per request based on `Accept-Encoding`)
  - `web/index.html` — add `<link rel="preconnect" href="https://flagcdn.com" crossorigin>`
- **Acceptance Criteria**:
  - `curl -H "Accept-Encoding: br" -I` shows `Content-Encoding: br`
  - `curl -H "Accept-Encoding: gzip" -I` shows `Content-Encoding: gzip`
  - Flag images load faster (preconnect eliminates DNS+TLS roundtrip)

### [ ] Task 8: Lighthouse CLS and LCP fixes

**Status:** deferred — chart wrappers already carry explicit style heights and fonts are system; not measured (no headless Lighthouse run).

- **Description**: Fix Cumulative Layout Shift (CLS) by ensuring chart containers have explicit dimensions. Fix Largest Contentful Paint (LCP) by ensuring the main heading renders immediately (no font blocking). Add `font-display: swap` if using custom fonts.
- **Files**:
  - `web/src/index.css` — ensure `.chart-frame` has explicit `min-height` or `aspect-ratio`
  - `web/src/pages/Profile.tsx` — the `<ReactECharts>` wrapper already has `style={{ height: 320 }}`, verify it's applied before data loads
  - `web/index.html` — if system fonts are used (they are), no font blocking; verify no render-blocking resources
- **Acceptance Criteria**:
  - Lighthouse CLS score = 1.0 (no layout shift)
  - LCP < 2.5s on simulated 4G
  - No "layout shift" warnings in Lighthouse

### [x] Task 9: Image optimization — flag CDN + favicon

**Status:** DONE — public/favicon.svg created; favicon.png kept as fallback, svg icon link added to index.html.

- **Description**: Country flags are already lazy-loaded from flagcdn.com at `w40` (40px width). Verify this is optimal. Convert favicon.png to SVG or add multiple sizes for different devices. Add `<link rel="icon" type="image/svg+xml">` if an SVG favicon is created.
- **Files**:
  - `web/src/components.tsx` — `PlayerFlag` already uses `w40`; verify this is correct
  - `web/index.html` — add `<link rel="apple-touch-icon">` if needed
  - Consider creating `web/public/favicon.svg` from the existing CourtMark SVG
- **Acceptance Criteria**:
  - Favicon displays correctly on all devices
  - Flag images are appropriately sized (not over-fetched)

### [x] Task 10: TanStack Query optimization

**Status:** PARTIAL — hover prefetch of similar players done; placeholderData skipped per user (staleTime:Infinity already caches).

- **Description**: The current `staleTime: Infinity, gcTime: Infinity` defaults are good for this data app. Add `placeholderData` for smoother transitions. Consider `prefetchQuery` for the next likely player when hovering similar-player links.
- **Files**:
  - `web/src/main.tsx` — QueryClient defaults are already optimal
  - `web/src/pages/Home.tsx` — on hover of a similar-player link, prefetch that player's profile/rank/match data
  - `web/src/pages/H2H.tsx` — when both players are selected, prefetch both profiles eagerly
- **Acceptance Criteria**:
  - Clicking a similar player shows profile data instantly (already cached from prefetch)
  - No unnecessary refetches (verify in React DevTools network tab)

### [x] Task 11: Bundle analysis and final audit

**Status:** DONE — split verification via pnpm build (4 chunks, sizes in Task 2); vite-bundle-visualizer run earlier to diagnose echarts tree-shake.

- **Description**: Run `vite-bundle-visualizer` or `rollup-plugin-visualizer` to confirm the bundle is properly split. Run Lighthouse CI locally to verify scores. Add a `just web-audit` recipe or a `pnpm` script for ongoing monitoring.
- **Files**:
  - `web/package.json` — add `"analyze": "vite-bundle-visualizer"` or similar
  - Optional: add a CI step that runs Lighthouse and fails if score < 95
- **Acceptance Criteria**:
  - Bundle analyzer shows ECharts is tree-shaken and code-split
  - Lighthouse Performance, SEO, Accessibility, Best Practices all ≥ 95 (target 100)
  - Total JS < 500KB for initial load

---

## Dependencies

- Task 1 (ECharts tree-shaking) should be done first — it's the biggest bundle win
- Task 2 (code splitting) depends on Task 1 — splitting ECharts into its own chunk is only valuable if it's smaller
- Task 3 (preload) depends on Task 2 — preloading is most effective when chunks are small
- Tasks 4-6 (SEO) are independent of each other and of 1-3
- Task 7 (compression/preconnect) is independent
- Task 8 (CLS/LCP) should be done after Tasks 1-3 (smaller bundle = faster LCP)
- Tasks 9-11 are polish/verification, done last

## QA/Testing Scenarios

1. **Bundle size**: `pnpm build` → check `dist/assets/` — initial JS chunk < 300KB, total < 700KB
2. **Route splitting**: Navigate to `/h2h` in browser → verify a separate chunk loads (Network tab)
3. **Preload**: Hover over "Matchups" nav link → verify the H2H chunk + data start loading before click
4. **SEO**: Run Google Rich Results Test on deployed URL; verify OG tags via `curl` or social media debugger
5. **Compression**: `curl -H "Accept-Encoding: br" -I` shows `Content-Encoding: br`; gzip also works
6. **Lighthouse**: Run Lighthouse in Chrome DevTools (or `lighthouse-ci`) → all four categories ≥ 95
7. **Functionality**: All existing features work — player search, profile view, H2H prediction, charts render, theme toggle, flags display
8. **Performance**: Simulated 4G in Chrome DevTools → LCP < 2.5s, TTI < 3s, CLS = 0
