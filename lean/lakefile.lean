import Lake
open Lake DSL

package «galois-ml-proofs» where
  -- Mathlib-free on purpose: the formalized statements are decidable
  -- arithmetic and structural inductions, so they check in seconds and
  -- carry no dependency risk. The probabilistic inequality behind the
  -- gate (Hoeffding / empirical Bernstein) is NOT formalized here.

lean_lib «GaloisMl» where
