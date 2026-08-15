import {
  createRootRoute,
  createRoute,
  HeadContent,
  lazyRouteComponent,
  Link,
  Outlet,
  useLocation,
} from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import Home from "./pages/Home";
import { formatLongDate } from "./lib/format";
import { usePlayerDirectory } from "./lib/playerIndex";
import { useTheme, type ThemeName } from "./theme";

function CourtMark() {
  return (
    <svg
      width="26"
      height="26"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <rect
        x="2.5"
        y="2.5"
        width="19"
        height="19"
        rx="3.5"
        stroke="var(--clay)"
        strokeWidth="2"
      />
      <path
        d="M2.5 12h19"
        stroke="var(--clay)"
        strokeWidth="1.4"
        opacity="0.65"
      />
      <path
        d="M12 2.5v19"
        stroke="var(--clay)"
        strokeWidth="1.4"
        opacity="0.65"
      />
      <circle cx="12" cy="12" r="1.6" fill="var(--clay)" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="8" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M2.8 2.8l1.1 1.1M12.1 12.1l1.1 1.1M13.2 2.8l-1.1 1.1M3.9 12.1l-1.1 1.1"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M13.4 9.4A5.6 5.6 0 0 1 6.6 2.6 5.6 5.6 0 1 0 13.4 9.4Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ThemeToggle({
  theme,
  onToggle,
}: {
  theme: ThemeName;
  onToggle: () => void;
}) {
  const dark = theme === "dark";
  return (
    <button
      type="button"
      className="theme-btn"
      onClick={onToggle}
      aria-pressed={!dark}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

function Layout() {
  const { theme, toggle } = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);
  const headerRef = useRef<HTMLElement>(null);
  const directoryQ = usePlayerDirectory();
  const { pathname } = useLocation();
  useEffect(() => {
    document.title =
      pathname === "/h2h" ? "Matchups — Courtside" : "Players — Courtside";
  }, [pathname]);
  // Close the mobile dropdown when the user taps/click outside the header or
  // presses Escape. A document-level listener is used instead of a viewport
  // scrim because the header's backdrop-filter creates a containing block for
  // fixed-position descendants, which trapped the old scrim inside the header.
  useEffect(() => {
    if (!menuOpen) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!headerRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);
  const profilesActive = pathname === "/";
  const h2hActive = pathname === "/h2h";

  return (
    <div className="app">
      <HeadContent />
      <header className="topnav" ref={headerRef}>
        <div className="container nav-inner">
          <Link
            to="/"
            search={{ player: undefined }}
            className="brand"
            aria-label="Courtside home"
            aria-expanded={menuOpen}
            aria-controls="nav-menu"
            onClick={(e) => {
              if (window.matchMedia("(max-width: 720px)").matches) {
                e.preventDefault();
                setMenuOpen((open) => !open);
              }
            }}
          >
            <CourtMark />
            <span className="brand-text">
              Courtside
              <span className="brand-kicker">Tennis Intelligence</span>
            </span>
            <svg
              className="brand-chevron"
              width="14"
              height="14"
              viewBox="0 0 16 16"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="m3.5 6 4.5 4.5L12.5 6"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </Link>
          <nav
            className={`nav-links${menuOpen ? " open" : ""}`}
            id="nav-menu"
            aria-label="Primary"
          >
            <Link
              to="/"
              search={{ player: undefined }}
              className={`navlink${profilesActive ? " active" : ""}`}
              aria-current={profilesActive ? "page" : undefined}
              onClick={() => setMenuOpen(false)}
            >
              Players
            </Link>
            <Link
              to="/h2h"
              search={{ playerA: undefined }}
              className={`navlink${h2hActive ? " active" : ""}`}
              aria-current={h2hActive ? "page" : undefined}
              onClick={() => setMenuOpen(false)}
            >
              H2H Matchups
            </Link>
          </nav>
          <div className="nav-actions">
            <ThemeToggle theme={theme} onToggle={toggle} />
            <a
              href="https://github.com/swang62/tennis-ml"
              target="_blank"
              rel="noopener noreferrer"
              className="theme-btn"
              aria-label="GitHub repository"
              title="GitHub repository"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
              </svg>
            </a>
          </div>
        </div>
      </header>
      <main className="container page">
        <Outlet />
      </main>
      <footer className="footer container">
        <span>Courtside — model-driven tennis intelligence.</span>
        {directoryQ.data?.latest_match_date && (
          <span>
            Last updated {formatLongDate(directoryQ.data.latest_match_date)}
          </span>
        )}
      </footer>
    </div>
  );
}

const rootRoute = createRootRoute({ component: Layout });

export const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Home,
  validateSearch: (search: Record<string, unknown>) => ({
    player:
      typeof search.player === "string"
        ? search.player
        : (undefined as string | undefined),
  }),
});

export const h2hRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/h2h",
  component: lazyRouteComponent(() => import("./pages/H2H")),
  validateSearch: (search: Record<string, unknown>) => ({
    playerA:
      typeof search.playerA === "string"
        ? search.playerA
        : (undefined as string | undefined),
  }),
});

export const routeTree = rootRoute.addChildren([homeRoute, h2hRoute]);