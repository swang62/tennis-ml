import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  type SyntheticEvent,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
import { homeRoute } from "../router";
import { useTheme } from "../theme";

// Profile (and its ECharts dependency) loads on demand once a player is
// selected, keeping the index chunk free of chart code.
const ProfileContent = lazy(() => import("./Profile"));

const PLAYERS_INDEX_KEY = "tm-player-index-v2";

const MINISEARCH_OPTS = {
  fields: ["display_name"],
  idField: "player_id",
  storeFields: [
    "display_name",
    "matches_played",
    "latest_rank_points",
    "current_rank",
    "ioc",
    "iso2",
    "country_name",
  ],
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
        latest_rank_points: r.latest_rank_points as number | undefined,
        current_rank: r.current_rank as number | null | undefined,
        ioc: r.ioc as string,
        iso2: r.iso2 as string,
        country_name: r.country_name as string,
      }));
  }, []);

  return { search, ready, loading: playersQ.isLoading && !ready };
}

export default function Home() {
  const { theme } = useTheme();
  const queryClient = useQueryClient();
  const navigate = homeRoute.useNavigate();
  const { player: searchPlayer } = homeRoute.useSearch();
  const selectedId = searchPlayer ?? null;
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
  // Directory rank backs the profile's current-rank label while the profile
  // query loads; the profile response is authoritative once it lands.
  const selectedPlayer =
    players.find((p) => p.player_id === selectedId) ?? null;

  // Route head sets the static "Players — Courtside"; once a player is picked
  // the selected name takes over (directory name backs the profile while it
  // loads). Reverts on deselect.
  const selectedName =
    profileQ.data?.display_name ?? selectedPlayer?.display_name ?? null;
  useEffect(() => {
    document.title = selectedName
      ? `${selectedName} — Courtside`
      : "Players — Courtside";
  }, [selectedName]);

  // Mirror Home's profile/rank/match/similar queries so hovering or focusing
  // a similar-player link makes the next selection render instantly.
  const prefetchPlayer = useCallback(
    (playerId: string) => {
      void queryClient.prefetchQuery({
        queryKey: ["profile", playerId],
        queryFn: () => getPlayerProfile(playerId),
      });
      void queryClient.prefetchQuery({
        queryKey: ["rank_history", playerId],
        queryFn: () => getRankHistory(playerId),
      });
      void queryClient.prefetchQuery({
        queryKey: ["match_history", playerId, 20],
        queryFn: () => getMatchHistory(playerId, 20),
      });
      void queryClient.prefetchQuery({
        queryKey: ["similar_players", playerId],
        queryFn: () => getSimilarPlayers(playerId, 3),
      });
    },
    [queryClient],
  );

  // Similar-player links render inside Profile; delegate from the container
  // and resolve the id via the already-loaded similar_players response.
  const handleSimilarPrefetch = useCallback(
    (e: SyntheticEvent) => {
      const btn = (e.target as HTMLElement).closest(".similar-link");
      if (!(btn instanceof HTMLElement)) return;
      const name = btn.textContent?.trim();
      if (!name) return;
      const sp = similarQ.data?.similar_players.find(
        (p) => p.display_name === name,
      );
      if (sp) prefetchPlayer(sp.player_id);
    },
    [similarQ.data, prefetchPlayer],
  );

  const handleSelectPlayer = (playerId: string | null) => {
    if (!document.startViewTransition) {
      navigate({
        to: "/",
        search: { player: playerId ?? undefined },
        replace: true,
      });
      return;
    }
    document.startViewTransition(() =>
      navigate({
        to: "/",
        search: { player: playerId ?? undefined },
        replace: true,
      }),
    );
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
          onMouseOver={handleSimilarPrefetch}
          onFocus={handleSimilarPrefetch}
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
            <Suspense fallback={<Loading label="Loading profile" />}>
              <ProfileContent
                profile={profileQ.data}
                directoryRank={selectedPlayer?.current_rank ?? null}
                rankHistory={rankQ.data}
                rankLoading={rankQ.isLoading}
                matchHistory={matchesQ.data}
                matchesLoading={matchesQ.isLoading}
                similarQ={similarQ}
                theme={theme}
                onSelectSimilar={handleSelectPlayer}
              />
            </Suspense>
          )}
        </div>
      )}
    </div>
  );
}
