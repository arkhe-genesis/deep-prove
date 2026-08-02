#!/bin/bash
set -e

cd arkhe-soc-tlm
BIN="./target/release/simulate_fountain"
cargo build --release --bin simulate_fountain

echo "Scenario 1: DSN (Low Loss - 1e-6)"
$BIN 256 0.000001 16 500

echo "Scenario 2: Interstellar (High Loss - 0.1)"
$BIN 256 0.1 16 500

echo "Scenario 3: Extreme Loss (0.99)"
$BIN 256 0.99 16 50000
