import { useState, type ReactNode } from 'react'
import type { Player } from './api'

// Format a 0..1 ratio as a percentage (e.g. 0.657 -> "65.7%").
export function pct(value: number | null): string {
  if (value == null) return 'n/a'
  return `${Math.round(value * 1000) / 10}%`
}

export function Kicker({ children }: { children: ReactNode }) {
  return <p className="kicker">{children}</p>
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="skeleton" role="status" aria-label={label}>
      <span className="sr-only">{label}...</span>
      <span className="skeleton-line" />
      <span className="skeleton-line" />
      <span className="skeleton-line short" />
    </div>
  )
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div className="error-box" role="alert">
      <p className="error-title">Failed to load</p>
      <p className="error-msg">{message}</p>
      {onRetry && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

export function Empty({ message }: { message: string }) {
  return <div className="empty">{message}</div>
}

export function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card">
      <h2 className="card-title">{title}</h2>
      {children}
    </section>
  )
}

export function StatBar({
  label,
  value,
  tone = 'grass',
}: {
  label: string
  value: number | null
  tone?: 'grass' | 'ice' | 'clay'
}) {
  const width = value == null ? 0 : Math.min(100, Math.max(0, value * 100))
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="text-[var(--text-dim)]">{label}</span>
        <span className="num font-semibold">{pct(value)}</span>
      </div>
      <div className="bar mt-1.5">
        <div className={`bar-fill is-${tone}`} style={{ width: `${width}%` }} />
      </div>
    </div>
  )
}

export function ResultBadge({ won }: { won: boolean }) {
  return (
    <span className={`badge ${won ? 'badge-grass' : 'badge-ice'}`}>{won ? 'Won' : 'Lost'}</span>
  )
}

// Compact W/L strip for recent form, oldest -> newest. Chips are immutable
// display items; key by result + occurrence count rather than list index.
export function FormStrip({ results }: { results: ('won' | 'lost')[] }) {
  const label = results.map((r) => (r === 'won' ? 'W' : 'L')).join(' ')
  const seen: Record<string, number> = {}
  return (
    <div className="form-strip" role="img" aria-label={`Recent form: ${label}`}>
      {results.map((r) => {
        const n = (seen[r] ?? 0) + 1
        seen[r] = n
        return (
          <span key={`${r}-${n}`} className={`form-chip ${r === 'won' ? 'is-win' : 'is-loss'}`}>
            {r === 'won' ? 'W' : 'L'}
          </span>
        )
      })}
    </div>
  )
}

// Searchable combobox over the players list (sorted by display_name upstream).
export function PlayerPicker({
  players,
  value,
  onChange,
  placeholder,
}: {
  players: Player[]
  value: string | null
  onChange: (playerId: string) => void
  placeholder: string
}) {
  const [query, setQuery] = useState('')
  const q = query.trim().toLowerCase()
  const filtered = players.filter(
    (p) =>
      p.display_name.toLowerCase().includes(q) || p.player_id.toLowerCase().includes(q),
  )
  const selected = players.find((p) => p.player_id === value)
  return (
    <div className="picker">
      <div className="picker-head">
        <span className="kicker">{placeholder}</span>
        {selected && <span className="picker-selected">{selected.display_name}</span>}
      </div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search players..."
        aria-label={`Search ${placeholder}`}
        className="input picker-input"
      />
      <div className="picker-list">
        {filtered.length === 0 && <div className="picker-empty">No matching players</div>}
        {filtered.map((p) => (
          <button
            key={p.player_id}
            type="button"
            onClick={() => {
              onChange(p.player_id)
              setQuery('')
            }}
            className={`picker-option ${p.player_id === value ? 'is-selected' : ''}`}
          >
            <span>{p.display_name}</span>
            <span className="picker-id">{p.player_id}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
