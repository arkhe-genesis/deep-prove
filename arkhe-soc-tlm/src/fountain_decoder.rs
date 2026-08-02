// Arkhe_Fountain_Decoder.rs
// SPDX-License-Identifier: MIT
// Selo: ARKHE-FOUNTAIN-DECODER-v1.0-2026-08-01

use std::collections::{HashMap, HashSet, VecDeque};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::fountain_encoder::{AFT_MAGIC, AFT_HEADER_SIZE, AFT_TRAILER_SIZE, OrchORState};

#[derive(Debug, Clone)]
pub struct FountainFrame {
    pub session_id: u32,
    pub seq_num: u32,
    pub k: usize,
    pub block_size: usize,
    pub degree: usize,
    pub seed: u32,
    pub payload: Vec<u8>,
}

impl FountainFrame {
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < AFT_HEADER_SIZE + 6 + AFT_TRAILER_SIZE {
            return None;
        }

        let magic = u32::from_le_bytes(data[0..4].try_into().ok()?);
        if magic != AFT_MAGIC {
            return None;
        }

        let payload_end = data.len() - AFT_TRAILER_SIZE;
        let expected_hash_bytes = &data[payload_end..];

        let hash = blake3::hash(&data[..payload_end]);
        if hash.as_bytes() != expected_hash_bytes {
            return None;
        }

        let session_id = u32::from_le_bytes(data[4..8].try_into().ok()?);
        let seq_num = u32::from_le_bytes(data[8..12].try_into().ok()?);
        let k = u16::from_le_bytes(data[12..14].try_into().ok()?) as usize;
        let block_size = u16::from_le_bytes(data[14..16].try_into().ok()?) as usize;
        let payload_len = u32::from_le_bytes(data[16..20].try_into().ok()?) as usize;
        let degree = u16::from_le_bytes(data[20..22].try_into().ok()?) as usize;
        let seed = u32::from_le_bytes(data[22..26].try_into().ok()?);

        let payload = data[26..26+payload_len].to_vec();
        Some(FountainFrame { session_id, seq_num, k, block_size, degree, seed, payload })
    }

    pub fn source_indices(&self) -> Vec<usize> {
        let mut rng = ChaCha20Rng::seed_from_u64(self.seed as u64);
        let mut selected = HashSet::with_capacity(self.degree);
        while selected.len() < self.degree {
            let idx = rng.gen_range(0..self.k);
            selected.insert(idx);
        }
        selected.into_iter().collect()
    }
}

pub struct FountainDecoder {
    pub decoded_blocks: HashMap<usize, Vec<u8>>,
    pub pending_frames: Vec<FountainFrame>,
    pub block_to_frames: HashMap<usize, Vec<usize>>,
    pub current_session: Option<u32>,
    pub expected_k: usize,
    pub block_size: usize,
}

impl FountainDecoder {
    pub fn new() -> Self {
        Self {
            decoded_blocks: HashMap::new(),
            pending_frames: Vec::new(),
            block_to_frames: HashMap::new(),
            current_session: None,
            expected_k: 0,
            block_size: 0,
        }
    }

    pub fn receive_frame(&mut self, raw_data: &[u8]) -> Result<bool, &'static str> {
        let frame = match FountainFrame::parse(raw_data) {
            Some(f) => f,
            None => return Err("Invalid frame or Hash mismatch"),
        };

        match self.current_session {
            None => {
                self.current_session = Some(frame.session_id);
                self.expected_k = frame.k;
                self.block_size = frame.block_size;
            }
            Some(sid) if sid != frame.session_id => {
                self.decoded_blocks.clear();
                self.pending_frames.clear();
                self.block_to_frames.clear();
                self.current_session = Some(frame.session_id);
                self.expected_k = frame.k;
                self.block_size = frame.block_size;
            }
            _ => {}
        }

        if self.is_complete() {
            return Ok(true);
        }

        let indices = frame.source_indices();

        let mut unresolved_indices = Vec::new();
        let mut resolved_xor = vec![0u8; frame.payload.len()];

        for &idx in &indices {
            if let Some(block) = self.decoded_blocks.get(&idx) {
                for (i, &byte) in block.iter().enumerate() {
                    resolved_xor[i] ^= byte;
                }
            } else {
                unresolved_indices.push(idx);
            }
        }

        let effective_degree = unresolved_indices.len();

        if effective_degree == 0 {
            return Ok(self.is_complete());
        }

        let mut effective_payload = frame.payload.clone();
        for (i, byte) in effective_payload.iter_mut().enumerate() {
            *byte ^= resolved_xor[i];
        }

        if effective_degree == 1 {
            let resolved_idx = unresolved_indices[0];
            self.decoded_blocks.insert(resolved_idx, effective_payload.clone());
            self.propagate_resolution(resolved_idx, &effective_payload);
            self.peel_cascade();
        } else {
            let frame_idx = self.pending_frames.len();
            self.pending_frames.push(FountainFrame {
                session_id: frame.session_id,
                seq_num: frame.seq_num,
                k: frame.k,
                block_size: frame.block_size,
                degree: effective_degree,
                seed: frame.seed,
                payload: effective_payload,
            });

            for &idx in &unresolved_indices {
                self.block_to_frames.entry(idx).or_insert_with(Vec::new).push(frame_idx);
            }
        }

        Ok(self.is_complete())
    }

    fn propagate_resolution(&mut self, resolved_idx: usize, resolved_data: &[u8]) {
        if let Some(frame_indices) = self.block_to_frames.remove(&resolved_idx) {
            for &frame_idx in &frame_indices {
                if frame_idx >= self.pending_frames.len() {
                    continue;
                }
                let frame = &mut self.pending_frames[frame_idx];
                for (i, byte) in frame.payload.iter_mut().enumerate() {
                    if i < resolved_data.len() {
                        *byte ^= resolved_data[i];
                    }
                }
                frame.degree -= 1;
            }
        }
    }

    fn peel_cascade(&mut self) {
        let mut queue: VecDeque<usize> = VecDeque::new();

        for (idx, frame) in self.pending_frames.iter().enumerate() {
            if frame.degree == 1 {
                queue.push_back(idx);
            }
        }

        while let Some(frame_idx) = queue.pop_front() {
            if frame_idx >= self.pending_frames.len() {
                continue;
            }
            let frame = &self.pending_frames[frame_idx];
            if frame.degree != 1 {
                continue;
            }

            let remaining_idx = self.find_remaining_block(frame_idx);
            if let Some(idx) = remaining_idx {
                if self.decoded_blocks.contains_key(&idx) {
                    continue;
                }
                let data = frame.payload.clone();
                self.decoded_blocks.insert(idx, data.clone());
                self.propagate_resolution(idx, &data);

                // Check for new degree-1 frames and add to queue
                for (i, f) in self.pending_frames.iter().enumerate() {
                    if f.degree == 1 && !queue.contains(&i) {
                        queue.push_back(i);
                    }
                }
            }
        }
    }

    fn find_remaining_block(&self, frame_idx: usize) -> Option<usize> {
        let frame = &self.pending_frames[frame_idx];
        let indices = frame.source_indices();
        for &idx in &indices {
            if !self.decoded_blocks.contains_key(&idx) {
                return Some(idx);
            }
        }
        None
    }

    pub fn is_complete(&self) -> bool {
        if self.expected_k == 0 {
            return false;
        }
        self.decoded_blocks.len() >= self.expected_k
    }

    pub fn reconstruct(&self) -> Option<Vec<u8>> {
        if !self.is_complete() {
            return None;
        }

        let mut result = Vec::new();
        for i in 0..self.expected_k {
            let block = self.decoded_blocks.get(&i)?;
            result.extend_from_slice(block);
        }
        Some(result)
    }

    pub fn reconstruct_orchor(&self) -> Option<OrchORState> {
        let data = self.reconstruct()?;
        OrchORState::from_bytes(&data)
    }

    pub fn progress(&self) -> f64 {
        if self.expected_k == 0 {
            return 0.0;
        }
        self.decoded_blocks.len() as f64 / self.expected_k as f64
    }
}

pub struct ErasureChannel {
    pub loss_rate: f64,
}

impl ErasureChannel {
    pub fn new(loss_rate: f64) -> Self {
        Self { loss_rate: loss_rate.clamp(0.0, 1.0) }
    }

    pub fn transmit<R: Rng>(&self, frame: &[u8], rng: &mut R) -> Option<Vec<u8>> {
        if rng.gen::<f64>() < self.loss_rate {
            None
        } else {
            Some(frame.to_vec())
        }
    }
}
