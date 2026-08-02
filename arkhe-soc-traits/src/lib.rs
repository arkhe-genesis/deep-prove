//! Arkhe SoC Traits — Hardware Interfaces and Memory Map
//! Selo: ARKHE-SOC-TRAITS-v25.0-2026-08-01

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocError {
    InvalidAddress(u32),
    AlignmentError(u32),
    Busy,
    SecurityViolation,
    Timeout,
}

pub trait ArkhePeripheral {
    fn read_reg(&mut self, addr: u32) -> Result<u32, SocError>;
    fn write_reg(&mut self, addr: u32, val: u32) -> Result<(), SocError>;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize)]
pub struct PerformanceCounters {
    pub qpl_cycles: u64,
    pub expand_cycles: u64,
    pub encode_cycles: u64,
    pub verify_cycles: u64,
    pub frames_emitted: u64,
    pub frames_dropped: u64,
    pub frames_rejected: u64,
    pub swarms_completed: u64,
    pub power_mw: u32,
    pub bus_contentions: u32,
}

impl PerformanceCounters {
    pub fn merge(&mut self, other: &Self) {
        self.qpl_cycles += other.qpl_cycles;
        self.expand_cycles += other.expand_cycles;
        self.encode_cycles += other.encode_cycles;
        self.verify_cycles += other.verify_cycles;
        self.frames_emitted += other.frames_emitted;
        self.frames_dropped += other.frames_dropped;
        self.frames_rejected += other.frames_rejected;
        self.swarms_completed += other.swarms_completed;
        self.bus_contentions += other.bus_contentions;
    }
}
