import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { getPlayers } from '../api'
import { Empty, ErrorBox, Kicker, Loading } from '../components'

type SortMode = 'name' | 'matches'

export default function Home() {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortMode>('name')
  const playersQ = useQuery({ queryKey: ['players'], queryFn: getPlayers })

  if (playersQ.isLoading) return <Loading label="Loading players" />
  if (playersQ.isError)
    return <ErrorBox error={playersQ.error} onRetry={() => playersQ.refetch()} />

  const players = playersQ.data?.players ?? []
  const q = query.trim().toLowerCase()
  const totalMatches = players.reduce((n, p) => n + p.matches_played, 0)
  const maxMatches = Math.max(1, ...players.map((p) => p.matches_played))

  const filtered = players
    .filter(
      (p) =>
        p.display_name.toLowerCase().includes(q) || p.player_id.toLowerCase().includes(q),
    )
    .sort((a, b) =>
      sort === 'matches'
        ? b.matches_played - a.matches_played
        : a.display_name.localeCompare(b.display_name),
    )

  return (
    <div>
      <section className="page-head">
        <Kicker>Player directory</Kicker>
        <h1 className="page-title">Players</h1>
        <p className="page-sub">
          The roster of ATP players tracked in the training corpus. Open a player for
          career form, surface splits, rank history and recent matches.
        </p>
        <div className="mt-5 flex flex-wrap gap-x-10 gap-y-4">
          <div className="stat">
            <span className="stat-label">Players</span>
            <span className="stat-num num">{players.length}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Matches</span>
            <span className="stat-num num">{totalMatches}</span>
          </div>
        </div>
      </section>

      <div className="toolbar">
        <div className="search">
          <svg
            className="search-icon"
            width="15"
            height="15"
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
          >
            <circle cx="7" cy="7" r="4.6" stroke="currentColor" strokeWidth="1.5" />
            <path d="M10.5 10.5 14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by name or id..."
            aria-label="Search players by name or id"
            className="input"
          />
        </div>
        <select
          className="select"
          style={{ maxWidth: '14rem' }}
          value={sort}
          onChange={(e) => setSort(e.target.value as SortMode)}
          aria-label="Sort players"
        >
          <option value="name">Sort: Name</option>
          <option value="matches">Sort: Most matches</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <Empty message="No players match your search" />
      ) : (
        <ul className="roster">
          {filtered.map((p, i) => (
            <li key={p.player_id}>
              <Link
                to="/players/$playerId"
                params={{ playerId: p.player_id }}
                className="roster-row"
              >
                <span className="roster-index num">{String(i + 1).padStart(2, '0')}</span>
                <span className="roster-main">
                  <span className="roster-name">{p.display_name}</span>
                  <span className="roster-id">{p.player_id}</span>
                </span>
                <span className="roster-matches">
                  <span className="roster-bar">
                    <span
                      className="roster-bar-fill"
                      style={{ width: `${Math.max(4, (p.matches_played / maxMatches) * 100)}%` }}
                    />
                  </span>
                  <span className="roster-count num">
                    {p.matches_played} {p.matches_played === 1 ? 'match' : 'matches'}
                  </span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
