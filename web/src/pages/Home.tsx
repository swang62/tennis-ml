import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { getPlayers } from '../api'
import { Empty, ErrorBox, Loading } from '../components'
export default function Home() {
  const [query, setQuery] = useState('')
  const playersQ = useQuery({ queryKey: ['players'], queryFn: getPlayers })

  if (playersQ.isLoading) return <Loading label="Loading players" />
  if (playersQ.isError)
    return <ErrorBox error={playersQ.error} onRetry={() => playersQ.refetch()} />

  const players = playersQ.data?.players ?? []
  const q = query.trim().toLowerCase()
  const filtered = players.filter(
    (p) =>
      p.display_name.toLowerCase().includes(q) || p.player_id.toLowerCase().includes(q),
  )

  return (
    <div>
      <h1 className="mb-1 text-2xl font-bold">Players</h1>
      <p className="mb-4 text-sm text-slate-600">
        Pick a player to open their profile ({players.length} players).
      </p>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search by name or id..."
        className="mb-4 w-full max-w-md rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-emerald-500"
      />
      {filtered.length === 0 && <Empty message="No players match your search" />}
      <ul className="grid gap-2 sm:grid-cols-2">
        {filtered.map((p) => (
          <li key={p.player_id}>
            <Link
              to="/players/$playerId"
              params={{ playerId: p.player_id }}
              className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm hover:border-emerald-400"
            >
              <span className="font-medium">{p.display_name}</span>
              <span className="text-xs text-slate-400">
                {p.player_id} - {p.matches_played} matches
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
