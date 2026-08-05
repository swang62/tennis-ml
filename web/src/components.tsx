import { useState, type ReactNode } from 'react'
import type { Player } from './api'

// Format a 0..1 ratio as a percentage (e.g. 0.657 -> "65.7%").
export function pct(value: number | null): string {
  if (value == null) return 'n/a'
  return `${Math.round(value * 1000) / 10}%`
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return <div className="py-8 text-center text-sm text-slate-500">{label}...</div>
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div className="rounded border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
      <p className="font-semibold">Failed to load</p>
      <p className="mt-1">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded bg-rose-600 px-3 py-1 text-xs font-medium text-white hover:bg-rose-700"
        >
          Retry
        </button>
      )}
    </div>
  )
}

export function Empty({ message }: { message: string }) {
  return <div className="py-8 text-center text-sm text-slate-500">{message}</div>
}

export function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
        {title}
      </h2>
      {children}
    </section>
  )
}

// Labeled progress bar, plain divs — no chart lib needed.
export function StatBar({ label, value }: { label: string; value: number | null }) {
  const width = value == null ? 0 : Math.min(100, Math.max(0, value * 100))
  return (
    <div>
      <div className="flex items-baseline justify-between text-sm">
        <span className="text-slate-600">{label}</span>
        <span className="font-medium">{pct(value)}</span>
      </div>
      <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full bg-emerald-500"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  )
}

export function ResultBadge({ won }: { won: boolean }) {
  return (
    <span
      className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${
        won ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'
      }`}
    >
      {won ? 'WON' : 'LOST'}
    </span>
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
    <div className="overflow-hidden rounded-lg border border-slate-300 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          {placeholder}
        </span>
        {selected && <span className="text-sm font-medium">{selected.display_name}</span>}
      </div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search players..."
        className="w-full border-b border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-500"
      />
      <div className="max-h-64 overflow-y-auto">
        {filtered.length === 0 && (
          <div className="px-3 py-2 text-sm text-slate-400">No matching players</div>
        )}
        {filtered.map((p) => (
          <button
            key={p.player_id}
            type="button"
            onClick={() => {
              onChange(p.player_id)
              setQuery('')
            }}
            className={`flex w-full items-center justify-between px-3 py-1.5 text-left text-sm hover:bg-slate-100 ${
              p.player_id === value ? 'bg-emerald-50' : ''
            }`}
          >
            <span>{p.display_name}</span>
            <span className="text-xs text-slate-400">{p.player_id}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
