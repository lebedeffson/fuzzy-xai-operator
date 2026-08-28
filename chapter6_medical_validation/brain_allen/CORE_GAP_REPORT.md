# Frozen-core gap: cross-XAI spatial Gamma

The public P19 runtime can build canonical Gamma from its generic model source
and registered target interface, but `ObservationContext` has no public typed
slot for injecting two experiment-side spatial `ExplanationObject` instances
plus their Pi/iota transforms. Therefore Grad-CAM/positive-IG agreement is
reported only as a separately named spatial diagnostic. It is not renamed
Gamma and no core change is made in Chapter 6. Spatial attribution reduction
is likewise `not_applied` with `w_Delta=0`.
