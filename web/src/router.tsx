import { createRootRoute, createRoute, createRouter, Link, Outlet, useLocation } from '@tanstack/react-router'
import Home from './pages/Home'
import H2H from './pages/H2H'
import { useTheme, type ThemeName } from './theme'

function CourtMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2.5" y="2.5" width="19" height="19" rx="3.5" stroke="var(--clay)" strokeWidth="2" />
      <path d="M2.5 12h19" stroke="var(--clay)" strokeWidth="1.4" opacity="0.65" />
      <path d="M12 2.5v19" stroke="var(--clay)" strokeWidth="1.4" opacity="0.65" />
      <circle cx="12" cy="12" r="1.6" fill="var(--clay)" />
    </svg>
  )
}

function SunIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M2.8 2.8l1.1 1.1M12.1 12.1l1.1 1.1M13.2 2.8l-1.1 1.1M3.9 12.1l-1.1 1.1"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path
        d="M13.4 9.4A5.6 5.6 0 0 1 6.6 2.6 5.6 5.6 0 1 0 13.4 9.4Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function ThemeToggle({ theme, onToggle }: { theme: ThemeName; onToggle: () => void }) {
  const dark = theme === 'dark'
  return (
    <button
      type="button"
      className="theme-btn"
      onClick={onToggle}
      aria-pressed={!dark}
      aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {dark ? <SunIcon /> : <MoonIcon />}
    </button>
  )
}

function Layout() {
  const { theme, toggle } = useTheme()
  const { pathname } = useLocation()
  const profilesActive = pathname === '/'
  const h2hActive = pathname === '/h2h'

  return (
    <div className="app">
      <header className="topnav">
        <div className="container nav-inner">
          <Link to="/" className="brand" aria-label="Courtside home">
            <CourtMark />
            <span className="brand-text">
              Courtside
              <span className="brand-kicker">Tennis Intelligence</span>
            </span>
          </Link>
          <nav className="nav-links" aria-label="Primary">
            <Link
              to="/"
              className={`navlink${profilesActive ? ' active' : ''}`}
              aria-current={profilesActive ? 'page' : undefined}
            >
              Player Profiles
            </Link>
            <Link
              to="/h2h"
              search={{ playerA: undefined }}
              className={`navlink${h2hActive ? ' active' : ''}`}
              aria-current={h2hActive ? 'page' : undefined}
            >
              Matchup Predictions
            </Link>
          </nav>
          <div className="nav-actions">
            <ThemeToggle theme={theme} onToggle={toggle} />
          </div>
        </div>
      </header>
      <main className="container page">
        <Outlet />
      </main>
      <footer className="footer container">
        <span>Courtside — model-driven tennis intelligence.</span>
        <span>Predictions are statistical model outputs, shown for information only; not betting advice.</span>
      </footer>
    </div>
  )
}

const rootRoute = createRootRoute({ component: Layout })

export const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: Home,
})

export const h2hRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/h2h',
  component: H2H,
  validateSearch: (search: Record<string, unknown>) => ({
    playerA: typeof search.playerA === 'string' ? search.playerA : undefined as string | undefined,
  }),
})

const routeTree = rootRoute.addChildren([homeRoute, h2hRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
