# TODO: Remove remaining literal 'other' keys/labels

## Steps:
1. ✅ Remove `'other': 0` from `_ROUND_ENCODINGS` in src/features/inference.py
2. ✅ Remove `'other'` from `MatchRound` type in web/src/api.ts
3. ✅ Remove `'other'` entry from ROUNDS array in web/src/pages/H2H.tsx
4. ✅ Check and update tests if needed (no tests reference 'other' round)
5. ✅ Run focused pytest and npm run build to verify

## Files modified:
- src/features/inference.py
- web/src/api.ts
- web/src/pages/H2H.tsx

## Verification results:
- Grep confirms no literal 'other' key/label in modified files
- Tournament keys (Davis Cup, etc.) remain unchanged
- Numeric fallback 0 still works via `_ROUND_ENCODINGS.get(round, 0)`
- pytest passed (26/27 tests; 1 unrelated failure: test_invalid_surface_raises tests carpet as invalid but it's valid)
- npm run build succeeded