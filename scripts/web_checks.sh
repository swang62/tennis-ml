#!/bin/sh
# Typecheck and unit-test the web dashboard. Catches the tsc errors (e.g.
# unused locals under noUnusedLocals) that otherwise only surface at
# `docker build` deploy time.
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT/web" || exit 1

pnpm exec tsc --noEmit || exit 1
pnpm test || exit 1
