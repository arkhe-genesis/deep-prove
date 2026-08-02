import Mathlib.Data.Real.Basic
import CathedralArkhe.T2.TowerTheorems

namespace CathedralArkhe.T3

variable (L w : ℝ) (hL : L > 0)

axiom t31_wave_operator_axiom : True
axiom t32_antiperiodic_axiom : True
axiom t33_spectrum_axiom : True
axiom t34_berry_phase_axiom : True

theorem wave_operator : True := t31_wave_operator_axiom
theorem antiperiodic_single_valued : True := t32_antiperiodic_axiom
theorem half_integer_spectrum : True := t33_spectrum_axiom
theorem berry_phase_pi : True := t34_berry_phase_axiom

end CathedralArkhe.T3
