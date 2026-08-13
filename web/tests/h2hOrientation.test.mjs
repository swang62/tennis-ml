import assert from "node:assert";
import test from "node:test";
import { orientH2H, preferenceEdge } from "../src/lib/h2hOrientation.ts";

const h2hResponse = {
  player1_id: "a",
  player2_id: "b",
  meetings: [{ match_id: "1", player1_won: true }],
  summary: {
    meetings: 1,
    player1_wins: 1,
    player2_wins: 0,
    player1_win_rate: 1,
    last5_player1_win_rate: 1,
  },
};

test("orients canonical H2H stats into Player A's picker position", () => {
  const oriented = orientH2H(h2hResponse, "b");

  assert.equal(oriented.player1_id, "b");
  assert.equal(oriented.player2_id, "a");
  assert.equal(oriented.meetings[0].player1_won, false);
  assert.deepEqual(oriented.summary, {
    meetings: 1,
    player1_wins: 0,
    player2_wins: 1,
    player1_win_rate: 0,
    last5_player1_win_rate: 0,
  });
});

test("preferenceEdge is negative when Player A is favored, positive when Player B is", () => {
  assert.ok(Math.abs(preferenceEdge(0.52) + 0.02) < 1e-10);
  assert.ok(Math.abs(preferenceEdge(0.48) - 0.02) < 1e-10);
  assert.equal(preferenceEdge(0.5), 0);
});
