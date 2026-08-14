/** Signed chart edge vs even: negative favors Player A (left), positive Player B (right). */
export function preferenceEdge(p: number): number {
  return 0.5 - p;
}
