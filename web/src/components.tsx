import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import type { Player } from './api'
import { pct, sanitizeErrorMessage } from './lib/format'

export function Kicker({ children }: { children: ReactNode }) {
  return <p className="kicker">{children}</p>
}

// Inline country flag from FlagCDN (20px wide). UNK/missing iso2 or a failed
// image load falls back to the native white-flag glyph, never a broken image.
export function PlayerFlag({
  iso2,
  countryName,
}: {
  iso2?: string | null
  countryName?: string | null
}) {
  const [failed, setFailed] = useState(false)
  const code = iso2?.trim().toLowerCase()
  const unknown = !code || code === 'unk' || code.length !== 2
  if (unknown || failed) {
    return (
      <span
        className="player-flag"
        role="img"
        aria-label="Country unknown"
        title="Country unknown"
      >
        🏳️
      </span>
    )
  }
  return (
    <img
      className="player-flag"
      src={`https://flagcdn.com/w40/${code}.png`}
      alt={countryName ?? code}
      title={countryName ?? code}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  )
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
    <span className={`badge ${won ? 'badge-grass' : 'badge-clay'}`}>{won ? 'Won' : 'Lost'}</span>
  )
}

// Oldest-to-newest result chips use stable result-occurrence keys.
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

// Closed-by-default combobox searches names but stores player ids.
export function PlayerPicker({
  players,
  value,
  onChange,
  placeholder,
  exclude,
  searchFn,
  loading,
  tone,
}: {
  players: Player[]
  value: string | null
  onChange: (playerId: string | null) => void
  placeholder: string
  exclude?: string | null
  searchFn?: (query: string) => Player[]
  loading?: boolean
  // Tints the trigger-row border (and focus outline) to the player color once
  // a player is selected; unselected pickers keep the default clay hover.
  tone?: "grass" | "clay"
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const uid = useId()

  const selected = players.find((p) => p.player_id === value) ?? null
  const q = query.trim().toLowerCase()
  // Unfiltered pickers show the ranked top-20 directory (the API already
  // orders by current_rank ASC NULLS LAST), with the other picker excluded.
  const rankedDefaults = players
    .filter((p) => p.player_id !== exclude)
    .slice(0, 20)
  const options =
    q.length === 0
      ? rankedDefaults
      : searchFn
        ? searchFn(q)
        : players.filter(
            (p) => p.player_id !== exclude && p.display_name.toLowerCase().includes(q),
          )
  const active = Math.min(activeIndex, Math.max(options.length - 1, 0))
  const showLoading = loading && (q.length > 0 || options.length === 0)

  const close = (refocus: boolean) => {
    setOpen(false)
    setQuery('')
    setActiveIndex(0)
    if (refocus) triggerRef.current?.focus()
  }

  // Focus on open and close on outside pointer presses.
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
      <div
        className={`picker-trigger-row${selected ? " is-selected" : ""}`}
        data-open={open || undefined}
        data-tone={tone || undefined}
      >
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
            {showLoading ? (
              <div className="picker-loading" role="status">
                Loading players...
              </div>
            ) : options.length === 0 ? (
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
                  <span className="picker-option-name">{p.display_name}</span>
                  {p.current_rank != null && (
                    <span className="picker-option-rank num">#{p.current_rank}</span>
                  )}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}
