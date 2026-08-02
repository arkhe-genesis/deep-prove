use crate::fountain_encoder::{encode_orchor_state, OrchORState};
use crate::fountain_decoder::{FountainDecoder, ErasureChannel};
use rand::thread_rng;

pub fn simulate(_k: usize, loss_rate: f64, block_size: usize, n_frames: usize) -> bool {
    let state = OrchORState {
        timestamp: 1722470400000000000,
        coherence_time: 8.33e-13,
        frequency: 1.2e12,
        energy: 7.95e-22,
        hexagon_state: [16384; 12],
        regime: 4,
    };

    let mut encoder = encode_orchor_state(&state, block_size);
    let channel = ErasureChannel::new(loss_rate);
    let mut decoder = FountainDecoder::new();
    let mut rng = thread_rng();

    for _ in 0..n_frames {
        let frame = encoder.next_frame();
        if let Some(received_frame) = channel.transmit(&frame, &mut rng) {
            if decoder.receive_frame(&received_frame).unwrap() {
                return true;
            }
        }
    }
    false
}
