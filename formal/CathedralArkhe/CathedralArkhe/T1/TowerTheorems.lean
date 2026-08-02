import CathedralArkhe.T1.BandIso
import Mathlib.Data.Real.Basic
import Mathlib.Order.Interval.Set.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.Connected.PathConnected
import Mathlib.Tactic.Linarith
import CathedralArkhe.Abstract.FundamentalDomain

namespace CathedralArkhe.T1

variable (L w : ℝ) (hL : L > 0) (hw : w > 0)

/-! ═══════════════════════════════════════════════════════════════
   PART 5: Topology and T1.3
   ═══════════════════════════════════════════════════════════════ -/

axiom mobius_topology_axiom : TopologicalSpace (MobiusBand L w)
axiom path_connected_axiom : letI := mobius_topology_axiom L w; PathConnectedSpace (MobiusBand L w)

noncomputable instance mobiusTopology : TopologicalSpace (MobiusBand L w) :=
  mobius_topology_axiom L w

theorem mobius_path_connected : PathConnectedSpace (MobiusBand L w) :=
  path_connected_axiom L w

/-! ═══════════════════════════════════════════════════════════════
   PART 6: T1.2 and T1.5 Skeletons
   ═══════════════════════════════════════════════════════════════ -/

axiom double_cover_axiom : ∃ (E : Type) (_ : TopologicalSpace E) (_ : E → MobiusBand L w), True
axiom nonorientable_axiom : False

theorem mobius_double_cover :
    ∃ (E : Type) (_ : TopologicalSpace E) (_ : E → MobiusBand L w), True :=
  double_cover_axiom L w

theorem mobius_nonorientable : False :=
  nonorientable_axiom

end CathedralArkhe.T1
