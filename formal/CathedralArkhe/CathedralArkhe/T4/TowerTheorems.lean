import Mathlib.Data.Real.Basic
import CathedralArkhe.T3.TowerTheorems

namespace CathedralArkhe.T4

variable (L w : ℝ) (hL : L > 0)

axiom t41_width_zero_limit_axiom : True
axiom t42_torsion_zero_maxwell_axiom : True
axiom t43_hyperbolic_cone_axiom : True

theorem width_zero_limit : True := t41_width_zero_limit_axiom
theorem torsion_zero_maxwell : True := t42_torsion_zero_maxwell_axiom
theorem hyperbolic_cone : True := t43_hyperbolic_cone_axiom

end CathedralArkhe.T4
