import { useState } from 'react'
import { createRootRoute, createRoute, createRouter, Link, Outlet } from '@tanstack/react-router'
import Home from './pages/Home'
import Profile from './pages/Profile'
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

function MenuIcon({ open }: { open: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      {open ? (
        <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      ) : (
        <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      )}
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
  const [menuOpen, setMenuOpen] = useState(false)
  const { theme, toggle } = useTheme()
  const closeMenu = () => setMenuOpen(false)

  return (
    <div className="app">
      <header className="topnav">
        <div className="container nav-inner">
          <Link to="/" className="brand" onClick={closeMenu}>
            <CourtMark />
            <span className="brand-text">
              Courtside
              <span className="brand-kicker">Tennis Intelligence</span>
            </span>
          </Link>
          <nav className="nav-links" aria-label="Primary">
            <Link
              to="/"
              className="navlink"
              activeProps={{ className: 'navlink active' }}
              onClick={closeMenu}
            >
              Players
            </Link>
            <Link
              to="/h2h"
              className="navlink"
              activeProps={{ className: 'navlink active' }}
              onClick={closeMenu}
            >
              Head-to-Head
            </Link>
          </nav>
          <div className="nav-actions">
            <ThemeToggle theme={theme} onToggle={toggle} />
            <button
              type="button"
              className="nav-toggle"
              aria-expanded={menuOpen}
              aria-controls="mobile-menu"
              aria-label="Toggle navigation menu"
              onClick={() => setMenuOpen((o) => !o)}
            >
              <MenuIcon open={menuOpen} />
            </button>
          </div>
        </div>
        <nav id="mobile-menu" className={`mobile-nav ${menuOpen ? 'is-open' : ''}`} aria-label="Mobile">
          <Link to="/" className="navlink" activeProps={{ className: 'navlink active' }} onClick={closeMenu}>
            Players
          </Link>
          <Link to="/h2h" className="navlink" activeProps={{ className: 'navlink active' }} onClick={closeMenu}>
            Head-to-Head
          </Link>
        </nav>
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

export const profileRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/players/$playerId',
  component: Profile,
})

export const h2hRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/h2h',
  component: H2H,
})

const routeTree = rootRoute.addChildren([homeRoute, profileRoute, h2hRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
