use arkhe_soc_tlm::fountain_simulation::simulate;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: {} <K> <loss_rate> <block_size> <n_frames>", args[0]);
        std::process::exit(1);
    }

    let k: usize = args[1].parse().unwrap();
    let loss_rate: f64 = args[2].parse().unwrap();
    let block_size: usize = args[3].parse().unwrap();
    let n_frames: usize = args[4].parse().unwrap();

    let success = simulate(k, loss_rate, block_size, n_frames);

    println!("{{\"success\": {}}}", success);
}
