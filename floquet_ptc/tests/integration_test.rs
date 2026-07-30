use floquet_ptc::exceptional_point::{ExceptionalPointResult, PTCSignature};
use floquet_ptc::floquet::FloquetHamiltonian;

#[test]
fn test_floquet_transitions() {
    let hamiltonian = FloquetHamiltonian {
        omega_0: 10.0,
        omega_d: 5.0,
        eta_0: 0.8, // Max modulation (80% mass)
        gamma_0: 0.1,
    };

    // Test around tau = 0 where eta = -0.8
    // |eta| = 0.8 > 0.1, so delta_gamma = sqrt(0.7) ~ 0.836
    // img_pos = 0.1 + 0.836 = 0.936
    // img_neg = 0.1 - 0.836 = -0.736
    // img_pos > img_neg, so we expect BrokenSymmetryGain

    let result_tau_0 = ExceptionalPointResult::analyze(&hamiltonian, 0.0);
    assert_eq!(result_tau_0.signature, PTCSignature::BrokenSymmetryGain);
    assert!(result_tau_0.loss_reduction_fraction > 0.5);

    // Find tau where effective_mass_modulation approaches -0.1
    // -0.8 * cos^2(5.0 * tau) = -0.1 => cos^2(5.0 * tau) = 1/8
    // 5.0 * tau = acos(sqrt(1/8))
    let tau_ep = (1.0f64 / 8.0f64).sqrt().acos() / 5.0;

    let result_ep = ExceptionalPointResult::analyze(&hamiltonian, tau_ep);
    assert_eq!(result_ep.signature, PTCSignature::Coalesced);
    assert_eq!(result_ep.loss_reduction_fraction, 0.0);
    assert_eq!(result_ep.gain_bandwidth_ghz, 40.0);
}
