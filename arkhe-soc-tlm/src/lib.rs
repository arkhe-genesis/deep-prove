//! Arkhe SoC TLM v2.0 — Transaction-Level Model
//! Selo: ARKHE-SOC-TLM-v2.0-2026-07-31

pub mod aotb;
pub mod power;
pub mod qpl;
pub mod soc;
pub mod sram;
pub mod fountain_encoder;
pub mod fountain_decoder;
pub mod fountain_simulation;

pub use arkhe_soc_traits::{SocError, ArkhePeripheral, PerformanceCounters};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const DOMAIN_NODES: usize = 8;
pub const FULL_NODES: usize = 16;
pub const MAX_SWARMS: usize = 15;
pub const SENTINEL: u64 = u64::MAX;
pub const REFERENCE_SOL_US: f64 = 2.37; // arXiv:2607.16100

// ==========================================================
// CLOCK DOMAIN (Simulação temporal)
// ==========================================================

pub struct ClockDomain {
    pub freq_mhz: u32,
    cycle_count: AtomicU64,
    _epoch: Instant,
}

impl ClockDomain {
    pub fn new(freq_mhz: u32) -> Self {
        Self {
            freq_mhz,
            cycle_count: AtomicU64::new(0),
            _epoch: Instant::now(),
        }
    }

    pub fn tick(&self, cycles: u64) -> Duration {
        self.cycle_count.fetch_add(cycles, Ordering::SeqCst);
        let ns = (cycles * 1_000) / self.freq_mhz as u64;
        Duration::from_nanos(ns)
    }

    pub fn cycles_to_us(&self, cycles: u64) -> f64 {
        (cycles as f64) / (self.freq_mhz as f64)
    }

    pub fn total_cycles(&self) -> u64 {
        self.cycle_count.load(Ordering::SeqCst)
    }
}

// ==========================================================
// TIPOS CORE DO DOMÍNIO
// ==========================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferId {
    Domain,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct QplResult {
    pub node: usize,
    pub input: f64,
    pub output: f64,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AotbFrame {
    pub session_id: [u8; 16],
    pub sequence: u64,
    pub nonce: u64,
    pub proof_hash: [u8; 32],
    pub domain_values: [f64; DOMAIN_NODES],
    pub weights: [u8; DOMAIN_NODES],
    pub signature: Vec<u8>, // tamanho fixo para determinismo de layout
}

impl AotbFrame {
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + 8 + 8 + 32 + 64 + 8);
        bytes.extend_from_slice(&self.session_id);
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&self.nonce.to_le_bytes());
        bytes.extend_from_slice(&self.proof_hash);
        for value in &self.domain_values {
            // Canonical big-endian para evitar divergência f64 LE/BE
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.extend_from_slice(&self.weights);
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    BadSignature,
    Replay,
    Sentinel,
    SessionMismatch,
    SequenceMismatch,
    BadNonce,
}

pub fn hash_state(domain: &[f64; DOMAIN_NODES]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for value in domain {
        hasher.update(value.to_be_bytes());
    }
    hasher.finalize().into()
}

pub fn micros(cycles: u64, freq_mhz: u32) -> f64 {
    (cycles as f64) / (freq_mhz as f64)
}
