# UsePartialRMSEqual and UseRstdScaleEqual predicates

Both `UsePartialRMSEqual` and `UseRstdScaleEqual` predicates are now implemented.
They are emitted by `ProblemType.predicates()` in `Contractions.py` (under the
`includeType` branch) and enforced by C++ structs in
`ContractionProblemPredicates.hpp`, registered in
`Serialization/ContractionPredicates.hpp`.

Each predicate compares the corresponding runtime problem flag
(`problem.usePartialRMS()` / `problem.useRstdScale()`) against the value baked
into the selected solution, preventing a partial-RMS or rstd-scale kernel from
being dispatched for an ordinary GEMM call even if logic files were inadvertently
mixed.
