import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import type { Player } from './api'
import { sanitizeErrorMessage } from './lib/format'

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

export function ErrorBox({
  error,
  onRetry,
  knownIds,
}: {
  error: unknown
  onRetry?: () => void
  knownIds?: readonly string[]
}) {
  const raw = error instanceof Error ? error.message : String(error)
  const message = knownIds ? sanitizeErrorMessage(raw, knownIds) : raw
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

// Closed-by-default, accessible combobox over the players list (sorted by
// display_name upstream). Search is by display name only; the selected value
// stays the player_id for state, query keys, routes and API calls.
export function PlayerPicker({
  players,
  value,
  onChange,
  placeholder,
  exclude,
}: {
  players: Player[]
  value: string | null
  onChange: (playerId: string | null) => void
  placeholder: string
  exclude?: string | null
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const uid = useId()

  const selected = players.find((p) => p.player_id === value)
  const q = query.trim().toLowerCase()
  const options = players.filter(
    (p) => p.player_id !== exclude && p.display_name.toLowerCase().includes(q),
  )
  const active = Math.min(activeIndex, Math.max(options.length - 1, 0))

  const close = (refocus: boolean) => {
    setOpen(false)
    setQuery('')
    setActiveIndex(0)
    if (refocus) triggerRef.current?.focus()
  }

  // While open, focus the search field and close on any pointer press outside
  // the widget, so the list is never left permanently expanded.
  useEffect(() => {
    if (!open) return
    inputRef.current?.focus()
    const onPointerDown = (e: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false)
        setQuery('')
        setActiveIndex(0)
      }
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open])

  const select = (playerId: string) => {
    onChange(playerId)
    close(true)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (options.length > 0) setActiveIndex((activeIndex + 1) % options.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (options.length > 0) setActiveIndex((activeIndex - 1 + options.length) % options.length)
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (options[active]) select(options[active].player_id)
    } else if (e.key === 'Escape') {
      e.preventDefault()
      close(true)
    } else if (e.key === 'Tab') {
      close(false)
    }
  }

  const triggerLabel = selected
    ? `${placeholder}, ${selected.display_name}`
    : `Select ${placeholder}`
  const listboxId = `${uid}-listbox`
  const optionId = (i: number) => `${uid}-option-${i}`

  return (
    <div className="picker" ref={rootRef}>
      <span className="picker-label">{placeholder}</span>
      <div className="picker-trigger-row" data-open={open || undefined}>
        <button
          ref={triggerRef}
          type="button"
          className={`picker-trigger${selected ? '' : ' is-empty'}`}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-label={triggerLabel}
          onClick={() => (open ? close(true) : setOpen(true))}
        >
          <span className="picker-trigger-name">
            {selected ? selected.display_name : 'Select player'}
          </span>
        </button>
        {selected && (
          <button
            type="button"
            className="picker-clear"
            aria-label={`Clear ${placeholder}`}
            onClick={() => {
              onChange(null)
              close(true)
            }}
          >
            ×
          </button>
        )}
        <span className="picker-caret" aria-hidden="true">
          ▾
        </span>
      </div>
      {open && (
        <div className="picker-popover">
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-controls={listboxId}
            aria-activedescendant={options.length > 0 ? optionId(active) : undefined}
            aria-autocomplete="list"
            aria-label={`Search ${placeholder}`}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setActiveIndex(0)
            }}
            onKeyDown={onKeyDown}
            placeholder="Search players..."
            className="input picker-input"
          />
          <div className="picker-list" role="listbox" id={listboxId}>
            {options.length === 0 ? (
              <div className="picker-empty" role="status">
                No matching players
              </div>
            ) : (
              options.map((p, i) => (
                <button
                  key={p.player_id}
                  type="button"
                  role="option"
                  id={optionId(i)}
                  aria-selected={p.player_id === value}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => select(p.player_id)}
                  className={`picker-option${i === active ? ' is-active' : ''}${p.player_id === value ? ' is-selected' : ''}`}
                >
                  {p.display_name}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}
