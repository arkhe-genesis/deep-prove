import Mathlib.Data.Real.Basic
import CathedralArkhe.Abstract.FundamentalDomain

namespace CathedralArkhe.T1

-- Real strip: S = ℝ × [-W/2, W/2]
def Strip (W : ℝ) := ℝ × { y : ℝ // -W/2 ≤ y ∧ y ≤ W/2 }

variable (L W : ℝ) (hL : L > 0) (hW : W > 0)

def mobiusRel (p q : Strip W) : Prop :=
  ∃ n : ℤ, p.1 + n * L = q.1 ∧ (if Even n then p.2.1 = q.2.1 else p.2.1 = -q.2.1)

axiom mobius_iseqv : Equivalence (mobiusRel L W)

def mobiusSetoid : Setoid (Strip W) where
  r := mobiusRel L W
  iseqv := mobius_iseqv L W

def MobiusBand := Quotient (mobiusSetoid L W)

end CathedralArkhe.T1
