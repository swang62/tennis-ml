import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  lazy,
  Suspense,
  type SyntheticEvent,
  useCallback,
  useEffect,
} from "react";
import {
  getMatchHistory,
  getPlayerProfile,
  getRankHistory,
  getSimilarPlayers,
} from "../api";
import { ErrorBox, Kicker, Loading, PlayerPicker } from "../components";
import { useDirectoryInfo } from "../lib/directoryInfo";
import { usePlayerDirectory } from "../lib/playerIndex";
import { homeRoute } from "../routes";
import { useTheme } from "../theme";

// Profile (and its ECharts dependency) loads on demand once a player is
// selected, keeping the index chunk free of chart code.
const ProfileContent = lazy(() => import("./Profile"));

export default function Home() {
  const { theme } = useTheme();
  const queryClient = useQueryClient();
  const navigate = homeRoute.useNavigate();
  const { player: searchPlayer } = homeRoute.useSearch();
  const selectedId = searchPlayer ?? null;
  const directoryQ = usePlayerDirectory();
  const players = directoryQ.data?.players ?? [];
  // True physical match total (distinct match_id), so a match is counted once
  // rather than once per participant. Shared with the Layout footer via one
  // idle-gated query; the match stat stays absent until the data resolves.
  const directoryInfoQ = useDirectoryInfo();
  // Directory rank backs the profile's current-rank label while the profile
  // query loads; the profile response is authoritative once it lands.
  const selectedPlayer =
    players.find((p) => p.player_id === selectedId) ?? null;

  // Queries below run only when a player is selected (enabled), so the id is
  // non-null there; assert it once instead of at every call site.
  const requireSelectedId = (): string => {
    if (selectedId === null) throw new Error("no player selected");
    return selectedId;
  };

  const profileQ = useQuery({
    queryKey: ["profile", selectedId],
    queryFn: () => getPlayerProfile(requireSelectedId()),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const rankQ = useQuery({
    queryKey: ["rank_history", selectedId],
    queryFn: () => getRankHistory(requireSelectedId()),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const matchesQ = useQuery({
    queryKey: ["match_history", selectedId, 20],
    queryFn: () => getMatchHistory(requireSelectedId(), 20),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const similarQ = useQuery({
    queryKey: ["similar_players", selectedId],
    queryFn: () => getSimilarPlayers(requireSelectedId(), 3),
    enabled: selectedId !== null,
    staleTime: Infinity,
    gcTime: Infinity,
  });

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
          Search for an ATP player to view career stats, surface splits, and
          match history.
        </p>
        {players.length > 0 && (
          <div className="mt-5 flex flex-wrap gap-x-10 gap-y-4">
            <div className="stat">
              <span className="stat-label">Players</span>
              <span className="stat-num num">{players.length}</span>
            </div>
            {directoryInfoQ.data && (
              <div className="stat">
                <span className="stat-label">Matches</span>
                <span className="stat-num num">
                  {directoryInfoQ.data.total_matches.toLocaleString("en-US")}
                </span>
              </div>
            )}
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
            searchLoader={directoryQ.data?.loadSearch}
            loading={directoryQ.isLoading}
          />
        </div>
        <Link
          to="/h2h"
          search={{
            playerA: selectedId ?? undefined,
            playerB: undefined,
          }}
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
        // biome-ignore lint/a11y/noStaticElementInteractions: container delegates mouse/focus prefetch to .similar-link child buttons
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
                directoryCluster={selectedPlayer?.cluster_label ?? null}
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
