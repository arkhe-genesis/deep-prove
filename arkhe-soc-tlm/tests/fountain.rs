use arkhe_soc_tlm::fountain_encoder::{encode_orchor_state, OrchORState};
use arkhe_soc_tlm::fountain_decoder::{FountainDecoder, ErasureChannel};
use rand::thread_rng;

#[test]
fn test_decoder_with_loss() {
    let state = OrchORState {
        timestamp: 1722470400000000000,
        coherence_time: 8.33e-13,
        frequency: 1.2e12,
        energy: 7.95e-22,
        hexagon_state: [16384; 12],
        regime: 4,
    };

    let mut encoder = encode_orchor_state(&state, 16);
    let channel = ErasureChannel::new(0.90);
    let mut decoder = FountainDecoder::new();
    let mut rng = thread_rng();

    let mut transmitted = 0;
    let mut received = 0;

    for _ in 0..50000 {
        let frame = encoder.next_frame();
        transmitted += 1;
        if let Some(received_frame) = channel.transmit(&frame, &mut rng) {
            received += 1;
            if decoder.receive_frame(&received_frame).unwrap() {
                break;
            }
        }
    }

    println!("Transmitted: {}, Received: {}, Progress: {:.1}%",
             transmitted, received, decoder.progress() * 100.0);

    assert!(decoder.is_complete());
    let reconstructed = decoder.reconstruct_orchor().unwrap();
    assert_eq!(state.timestamp, reconstructed.timestamp);
    assert_eq!(state.frequency, reconstructed.frequency);
}
