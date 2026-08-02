// Arkhe_Fountain_Encoder.rs
// SPDX-License-Identifier: MIT
// Selo: ARKHE-FOUNTAIN-ENCODER-v1.0-2026-08-01

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand::rngs::OsRng;

pub const AFT_HEADER_SIZE: usize = 20;
pub const AFT_TRAILER_SIZE: usize = 32; // Blake3
pub const AFT_MAGIC: u32 = 0x41525448;

#[derive(Debug, Clone)]
pub struct OrchORState {
    pub timestamp: u64,
    pub coherence_time: f64,
    pub frequency: f64,
    pub energy: f64,
    pub hexagon_state: [u16; 12],
    pub regime: u8,
}

impl OrchORState {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.coherence_time.to_le_bytes());
        buf.extend_from_slice(&self.frequency.to_le_bytes());
        buf.extend_from_slice(&self.energy.to_le_bytes());
        for &v in &self.hexagon_state {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.push(self.regime);
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 57 {
            return None;
        }
        let mut offset = 0;
        let timestamp = u64::from_le_bytes(bytes[offset..offset+8].try_into().ok()?);
        offset += 8;
        let coherence_time = f64::from_le_bytes(bytes[offset..offset+8].try_into().ok()?);
        offset += 8;
        let frequency = f64::from_le_bytes(bytes[offset..offset+8].try_into().ok()?);
        offset += 8;
        let energy = f64::from_le_bytes(bytes[offset..offset+8].try_into().ok()?);
        offset += 8;
        let mut hexagon_state = [0u16; 12];
        for i in 0..12 {
            hexagon_state[i] = u16::from_le_bytes(bytes[offset..offset+2].try_into().ok()?);
            offset += 2;
        }
        let regime = bytes[offset];
        Some(OrchORState {
            timestamp, coherence_time, frequency, energy, hexagon_state, regime,
        })
    }
}

pub struct RobustSoliton {
    pub k: usize,
    pub c: f64,
    pub delta: f64,
    cdf: Vec<f64>,
}

impl RobustSoliton {
    pub fn new(k: usize, c: f64, delta: f64) -> Self {
        let r = (c * (k as f64) / (k as f64).ln()).ceil() as usize;
        let mut rho = vec![0.0; k + 1];
        let mut tau = vec![0.0; k + 1];

        if k > 0 {
            rho[1] = 1.0 / (k as f64);
            for d in 2..=k {
                rho[d] = 1.0 / ((d * (d - 1)) as f64);
            }
        }

        if r > 0 {
            for d in 1..=(k / r).saturating_sub(1) {
                tau[d] = 1.0 / ((d * r) as f64);
            }
            if k >= r {
                tau[k / r] = (r as f64) * (k as f64).ln() / (k as f64);
            }
        }

        let z: f64 = (1..=k).map(|d| rho[d] + tau[d]).sum();
        let mut cdf = vec![0.0; k + 1];
        if z > 0.0 {
            let mut acc = 0.0;
            for d in 1..=k {
                acc += (rho[d] + tau[d]) / z;
                cdf[d] = acc;
            }
            cdf[k] = 1.0;
        }

        Self { k, c, delta, cdf }
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> usize {
        if self.k <= 1 {
            return self.k;
        }
        let u: f64 = rng.gen();
        let mut lo = 1usize;
        let mut hi = self.k;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cdf[mid] < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

pub struct FountainEncoder {
    pub blocks: Vec<Vec<u8>>,
    pub k: usize,
    pub block_size: usize,
    pub soliton: RobustSoliton,
    pub session_id: u32,
    pub seq_num: u32,
    pub rng: ChaCha20Rng,
}

impl FountainEncoder {
    pub fn new(data: &[u8], block_size: usize, c: f64, delta: f64) -> Self {
        let k = (data.len() + block_size - 1) / block_size;
        let mut blocks = Vec::with_capacity(k);
        for i in 0..k {
            let start = i * block_size;
            let end = (start + block_size).min(data.len());
            let mut block = data[start..end].to_vec();
            block.resize(block_size, 0);
            blocks.push(block);
        }

        let soliton = RobustSoliton::new(k, c, delta);

        let mut os_rng = OsRng;
        let session_id = os_rng.gen();
        let rng = ChaCha20Rng::from_rng(os_rng).unwrap();

        Self { blocks, k, block_size, soliton, session_id, seq_num: 0, rng }
    }

    pub fn next_frame(&mut self) -> Vec<u8> {
        let d = self.soliton.sample(&mut self.rng);
        let seed = self.seq_num.wrapping_mul(0x9E3779B9);
        let mut block_rng = ChaCha20Rng::seed_from_u64(seed as u64);

        let mut selected = Vec::with_capacity(d);
        while selected.len() < d {
            let idx = block_rng.gen_range(0..self.k);
            if !selected.contains(&idx) {
                selected.push(idx);
            }
        }

        let mut payload = vec![0u8; self.block_size];
        for &idx in &selected {
            for (i, byte) in self.blocks[idx].iter().enumerate() {
                payload[i] ^= byte;
            }
        }

        let mut frame = Vec::with_capacity(AFT_HEADER_SIZE + 6 + payload.len() + AFT_TRAILER_SIZE);
        frame.extend_from_slice(&AFT_MAGIC.to_le_bytes());
        frame.extend_from_slice(&self.session_id.to_le_bytes());
        frame.extend_from_slice(&self.seq_num.to_le_bytes());
        frame.extend_from_slice(&(self.k as u16).to_le_bytes());
        frame.extend_from_slice(&(self.block_size as u16).to_le_bytes());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&(d as u16).to_le_bytes());
        frame.extend_from_slice(&seed.to_le_bytes());
        frame.extend_from_slice(&payload);

        let hash = blake3::hash(&frame);
        frame.extend_from_slice(hash.as_bytes());

        self.seq_num = self.seq_num.wrapping_add(1);
        frame
    }

    pub fn generate_frames(&mut self, n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|_| self.next_frame()).collect()
    }
}

pub fn encode_orchor_state(state: &OrchORState, block_size: usize) -> FountainEncoder {
    let data = state.to_bytes();
    FountainEncoder::new(&data, block_size, 0.03, 0.5)
}
