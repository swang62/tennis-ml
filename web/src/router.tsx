import { createRootRoute, createRoute, createRouter, Link, Outlet } from '@tanstack/react-router'
import Home from './pages/Home'
import Profile from './pages/Profile'
import H2H from './pages/H2H'

function Layout() {
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <nav className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
          <Link to="/" className="text-lg font-bold tracking-tight">
            Tennis ML
          </Link>
          <Link
            to="/"
            className="text-sm text-slate-600 hover:text-slate-900 [&.active]:font-semibold [&.active]:text-slate-900"
          >
            Players
          </Link>
          <Link
            to="/h2h"
            className="text-sm text-slate-600 hover:text-slate-900 [&.active]:font-semibold [&.active]:text-slate-900"
          >
            Head-to-Head
          </Link>
        </nav>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
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
