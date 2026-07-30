//! Estruturas básicas para modelar a cavidade plasmônica
//! (Placeholder for completeness based on the Nature 2026 paper)

#[derive(Debug, Clone)]
pub struct PlasmonicCavity {
    pub resonance_frequency: f64, // THz
    pub quality_factor: f64,
}

#[derive(Debug, Clone)]
pub struct CarrierMassModulation {
    pub base_mass: f64,
    pub max_modulation_depth: f64, // up to 0.8 according to paper
}
