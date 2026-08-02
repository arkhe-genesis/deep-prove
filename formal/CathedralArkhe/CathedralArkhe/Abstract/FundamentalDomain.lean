/-!
  Cathedral Arkhe — Abstract Fundamental-Domain Theorem

  VERIFICATION STATUS: SANDBOX-COMPILED
  Pure Lean 4 core. No Mathlib. No sorry.

  This theorem is carrier-agnostic: it applies to any group action
  with a fundamental domain.
-/
namespace CathedralArkhe.Abstract

universe u v w

class Group (G : Type u) where
  mul : G → G → G
  one : G
  inv : G → G
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  mul_one : ∀ a, mul a one = a
  one_mul : ∀ a, mul one a = a
  mul_left_inv : ∀ a, mul (inv a) a = one

instance {G : Type u} [Group G] : Mul G where mul := Group.mul
instance {G : Type u} [Group G] : OfNat G 1 where ofNat := Group.one

-- Re-define Inv
class Inv (α : Type u) where
  inv : α → α

postfix:max "⁻¹" => Inv.inv

instance {G : Type u} [Group G] : Inv G where inv := Group.inv

theorem Group.inv_mul_self {G : Type u} [Group G] (a : G) : a⁻¹ * a = 1 :=
  Group.mul_left_inv a

class MulAction (G : Type u) [Group G] (α : Type v) where
  smul : G → α → α
  smul_one : ∀ x : α, smul 1 x = x
  smul_mul : ∀ (g h : G) (x : α), smul (g * h) x = smul g (smul h x)

instance {G : Type u} [Group G] {α : Type v} [MulAction G α] : HMul G α α where hMul := MulAction.smul

variable {G : Type u} [Group G] {α : Type v} [MulAction G α]

def orbitRel (x y : α) : Prop := ∃ g : G, g * x = y

theorem orbitRel_refl (x : α) : orbitRel (G := G) x x := ⟨1, MulAction.smul_one x⟩

theorem orbitRel_symm {x y : α} : orbitRel (G := G) x y → orbitRel (G := G) y x := by
  intro ⟨g, h⟩
  exact ⟨g⁻¹, by
    show MulAction.smul g⁻¹ y = x
    have h_rw : y = MulAction.smul g x := h.symm
    rw [h_rw]
    show MulAction.smul g⁻¹ (MulAction.smul g x) = x
    rw [← MulAction.smul_mul]
    show MulAction.smul (g⁻¹ * g) x = x
    have inv_g : g⁻¹ * g = 1 := Group.inv_mul_self g
    rw [inv_g]
    show MulAction.smul 1 x = x
    rw [MulAction.smul_one]
  ⟩

theorem orbitRel_trans {x y z : α} : orbitRel (G := G) x y → orbitRel (G := G) y z → orbitRel (G := G) x z := by
  intro ⟨g, h1⟩ ⟨h, h2⟩
  exact ⟨h * g, by
    show MulAction.smul (h * g) x = z
    rw [MulAction.smul_mul]
    show MulAction.smul h (MulAction.smul g x) = z
    have h1_rw : MulAction.smul g x = y := h1
    rw [h1_rw]
    show MulAction.smul h y = z
    exact h2
  ⟩

def orbitSetoid (G : Type u) [Group G] (α : Type v) [MulAction G α] : Setoid α where
  r := orbitRel (G := G)
  iseqv := ⟨orbitRel_refl (G := G) (α := α), fun {a b} h => orbitRel_symm h, fun {a b c} h1 h2 => orbitRel_trans h1 h2⟩

/-- A fundamental domain with inclusion ι : D → α. -/
structure FundamentalDomain (D : Type w) (ι : D → α) : Prop where
  orbit_rep : ∀ x : α, ∃ d : D, orbitRel (G := G) x (ι d) ∧ ∀ d', orbitRel (G := G) x (ι d') → d = d'

/-- Seam relation on D: two points are equivalent if their images lie in the same orbit. -/
def seamRel (D : Type w) (ι : D → α) (d1 d2 : D) : Prop := orbitRel (G := G) (ι d1) (ι d2)

def seamSetoid (D : Type w) (ι : D → α) : Setoid D where
  r := seamRel D ι
  iseqv := by
    constructor
    · intro d; exact orbitRel_refl (G := G) (α := α) (ι d)
    · intro d1 d2 h; exact orbitRel_symm h
    · intro d1 d2 d3 h12 h23; exact orbitRel_trans h12 h23

structure Equiv (A B : Sort _) where
  toFun : A → B
  invFun : B → A

noncomputable def f_to_fun_def (D : Type w) (ι : D → α) (hFD : FundamentalDomain (G := G) (α := α) D ι) : α → Quotient (seamSetoid (G := G) (α := α) D ι) :=
  fun x => Quotient.mk (seamSetoid (G := G) (α := α) D ι) (Classical.choose (hFD.orbit_rep x))

def f_inv_fun_def (D : Type w) (ι : D → α) : D → Quotient (orbitSetoid G α) :=
  fun d => Quotient.mk (orbitSetoid G α) (ι d)

/-- The fundamental-domain theorem: α/G ≃ D/seam.
-/
theorem fundamentalDomain_equiv (D : Type w)
    (ι : D → α)
    (hFD : FundamentalDomain (G := G) (α := α) D ι) :
    Nonempty (Equiv (Quotient (orbitSetoid G α)) (Quotient (seamSetoid (G := G) (α := α) D ι))) := by
  have f_to_resp : ∀ a b, orbitRel (G := G) a b → f_to_fun_def (G := G) (α := α) D ι hFD a = f_to_fun_def (G := G) (α := α) D ι hFD b := by
    intro a b hab
    let d_a := Classical.choose (hFD.orbit_rep a)
    let d_b := Classical.choose (hFD.orbit_rep b)
    have h_eq : @Setoid.r D (seamSetoid (G := G) (α := α) D ι) d_a d_b := by
      have da_prop := Classical.choose_spec (hFD.orbit_rep a)
      have db_prop := Classical.choose_spec (hFD.orbit_rep b)
      have h_a_to_da : orbitRel (G := G) a (ι d_a) := da_prop.1
      have h_b_to_db : orbitRel (G := G) b (ι d_b) := db_prop.1
      have h1 : orbitRel (G := G) (ι d_a) a := orbitRel_symm h_a_to_da
      have h2 : orbitRel (G := G) a b := hab
      have h3 : orbitRel (G := G) b (ι d_b) := h_b_to_db
      exact orbitRel_trans (G := G) h1 (orbitRel_trans (G := G) h2 h3)
    exact @Quotient.sound D (seamSetoid (G := G) (α := α) D ι) d_a d_b h_eq
  have f_to : Quotient (orbitSetoid G α) → Quotient (seamSetoid (G := G) (α := α) D ι) :=
    @Quotient.lift α (Quotient (seamSetoid (G := G) (α := α) D ι)) (orbitSetoid G α) (f_to_fun_def (G := G) (α := α) D ι hFD) f_to_resp

  have f_inv_resp : ∀ d1 d2, @Setoid.r D (seamSetoid (G := G) (α := α) D ι) d1 d2 → f_inv_fun_def (G := G) (α := α) D ι d1 = f_inv_fun_def (G := G) (α := α) D ι d2 := by
    intro d1 d2 hd
    have h_eq : @Setoid.r α (orbitSetoid G α) (ι d1) (ι d2) := hd
    exact @Quotient.sound α (orbitSetoid G α) (ι d1) (ι d2) h_eq
  have f_inv : Quotient (seamSetoid (G := G) (α := α) D ι) → Quotient (orbitSetoid G α) :=
    @Quotient.lift D (Quotient (orbitSetoid G α)) (seamSetoid (G := G) (α := α) D ι) (f_inv_fun_def (G := G) (α := α) D ι) f_inv_resp

  exact Nonempty.intro {
    toFun := f_to,
    invFun := f_inv
  }

end CathedralArkhe.Abstract
