import assert from "node:assert";
import test from "node:test";
import {
  orientH2H,
  pickerPreferenceEdge,
  probabilityForPlayer,
} from "../src/lib/h2hOrientation.ts";

const canonical = {
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
  const oriented = orientH2H(canonical, "b");

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

test("keeps a canonical model probability attached to the same player", () => {
  assert.equal(probabilityForPlayer(0.2, "alcaraz", "alcaraz"), 0.2);
  assert.equal(probabilityForPlayer(0.2, "alcaraz", "sinner"), 0.8);
});

test("maps canonical probability edge to left Player A and right Player B", () => {
  assert.ok(Math.abs(pickerPreferenceEdge(0.8, "alcaraz", "alcaraz") + 0.3) < 1e-10);
  assert.ok(Math.abs(pickerPreferenceEdge(0.8, "alcaraz", "sinner") - 0.3) < 1e-10);
});
