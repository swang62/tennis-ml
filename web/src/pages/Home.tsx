import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import MiniSearch from "minisearch";
import {
  getMatchHistory,
  getPlayerProfile,
  getPlayers,
  getRankHistory,
  getSimilarPlayers,
  type Player,
} from "../api";
import { ErrorBox, Kicker, Loading, PlayerPicker } from "../components";
import { useTheme } from "../theme";
import ProfileContent from "./Profile";

const PLAYERS_INDEX_KEY = "tm-player-index-v1";

const MINISEARCH_OPTS = {
  fields: ["display_name"],
  idField: "player_id",
  storeFields: ["display_name", "matches_played"],
  searchOptions: { fuzzy: 0.2, prefix: true, boost: { display_name: 2 } },
};

function useMiniSearch() {
  const indexRef = useRef<MiniSearch | null>(null);
  const [ready, setReady] = useState(false);

  const playersQ = useQuery({
    queryKey: ["players"],
    queryFn: getPlayers,
    staleTime: Infinity,
    gcTime: Infinity,
  });

  useEffect(() => {
    try {
      const cached = localStorage.getItem(PLAYERS_INDEX_KEY);
      if (cached) {
        indexRef.current = MiniSearch.loadJSON(cached, MINISEARCH_OPTS);
        setReady(true);
      }
    } catch {
      localStorage.removeItem(PLAYERS_INDEX_KEY);
    }
  }, []);

  useEffect(() => {
    if (!playersQ.data || indexRef.current) return;
    const ms = new MiniSearch(MINISEARCH_OPTS);
    ms.addAll(playersQ.data.players as any);
    indexRef.current = ms;
    setReady(true);
    try {
      localStorage.setItem(PLAYERS_INDEX_KEY, JSON.stringify(ms));
    } catch {
      // localStorage full, non-critical
    }
  }, [playersQ.data]);

  const search = useCallback((query: string): Player[] => {
    if (!indexRef.current || !query.trim()) return [];
    return indexRef.current
      .search(query.trim(), { fuzzy: 0.2, prefix: true })
      .map((r) => ({
        player_id: r.id,
        display_name: r.display_name as string,
        matches_played: r.matches_played as number,
      }));
  }, []);

  return { search, ready, loading: playersQ.isLoading && !ready };
}

export default function Home() {
  const { theme } = useTheme();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { search, ready, loading } = useMiniSearch();

  const profileQ = useQuery({
    queryKey: ["profile", selectedId],
    queryFn: () => getPlayerProfile(selectedId!),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const rankQ = useQuery({
    queryKey: ["rank_history", selectedId],
    queryFn: () => getRankHistory(selectedId!),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const matchesQ = useQuery({
    queryKey: ["match_history", selectedId, 20],
    queryFn: () => getMatchHistory(selectedId!, 20),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const similarQ = useQuery({
    queryKey: ["similar_players", selectedId],
    queryFn: () => getSimilarPlayers(selectedId!, 3),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });

  const playersQ = useQuery({
    queryKey: ["players"],
    queryFn: getPlayers,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const players = playersQ.data?.players ?? [];
  const totalMatches = players.reduce((n, p) => n + p.matches_played, 0);

  const handleSelectPlayer = (playerId: string | null) => {
    if (!document.startViewTransition) {
      setSelectedId(playerId);
      return;
    }
    document.startViewTransition(() => setSelectedId(playerId));
  };

  return (
    <div>
      <section className="page-head">
        <Kicker>Player directory</Kicker>
        <h1 className="page-title">Players</h1>
        <p className="page-sub">
          Search for an ATP player to view career stats, surface splits, rank
          history and recent matches.
        </p>
        {players.length > 0 && (
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
        )}
      </section>

      <div className="toolbar">
        <div className="toolbar-picker">
          <PlayerPicker
            players={players}
            value={selectedId}
            onChange={handleSelectPlayer}
            placeholder="Player"
            searchFn={ready ? search : undefined}
            loading={loading}
          />
        </div>
        <Link
          to="/h2h"
          search={{ playerA: selectedId ?? undefined } as any}
          className={`toolbar-compare${selectedId ? "" : " disabled"}`}
          aria-disabled={!selectedId}
          tabIndex={selectedId ? 0 : -1}
        >
          Predict H2H
        </Link>
      </div>

      {selectedId === null && (
        <div className="empty" style={{ marginTop: "1.25rem" }}>
          Select a player to view their profile
        </div>
      )}

      {selectedId !== null && (
        <div
          id="profile-anchor"
          className="mt-8"
          style={{ viewTransitionName: "profile-content" }}
        >
          {profileQ.isLoading && <Loading label="Loading profile" />}
          {profileQ.isError && (
            <ErrorBox
              error={profileQ.error}
              onRetry={() => profileQ.refetch()}
              knownIds={[selectedId]}
            />
          )}
          {profileQ.data && (
            <ProfileContent
              profile={profileQ.data}
              rankHistory={rankQ.data}
              rankLoading={rankQ.isLoading}
              matchHistory={matchesQ.data}
              matchesLoading={matchesQ.isLoading}
              similarQ={similarQ}
              theme={theme}
              onSelectSimilar={handleSelectPlayer}
            />
          )}
        </div>
      )}
    </div>
  );
}
