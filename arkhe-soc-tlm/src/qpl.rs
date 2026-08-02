use crate::{
    ArkhePeripheral, ClockDomain, DOMAIN_NODES, PerformanceCounters, QplResult, SocError,
};
use crate::sram::SramDxController;

#[repr(C)]
pub struct QplAcceleratorConfig {
    pub control: u32,    // bit 0: start, bit 1: reset
    pub status: u32,    // bit 0: done, bit 1: error
    pub iterations: u32,
    pub threshold: u32,  // veto threshold (focal_weight scaled)
}

pub struct QplAccelerator {
    pub config: QplAcceleratorConfig,
    pub counters: PerformanceCounters,
    clock: ClockDomain,
    _latency_model_us: f64, // meta: 2.37us para 8 nós (arXiv)
}

impl QplAccelerator {
    pub fn new(clock: ClockDomain) -> Self {
        Self {
            config: QplAcceleratorConfig {
                control: 0,
                status: 0,
                iterations: 0,
                threshold: 0,
            },
            counters: PerformanceCounters::default(),
            clock,
            _latency_model_us: 2.37,
        }
    }

    /// Executa 1 hop de QPL (convolution de vizinhança)
    pub fn execute_convolution(
        &mut self,
        sram: &SramDxController,
    ) -> Result<[QplResult; DOMAIN_NODES], SocError> {
        if self.config.control & 0x1 == 0 {
            return Err(SocError::Busy);
        }

        let start_cycles = self.clock.total_cycles();

        // Sentinel polling: aguarda SRAM ready
        let ready_seq = self.config.iterations as u64;
        while !sram.is_ready(ready_seq) {
            if self.counters.bus_contentions >= 1000 { break; }
            self.counters.bus_contentions += 1;
        }

        let mut results = [QplResult {
            node: 0,
            input: 0.0,
            output: 0.0,
        }; DOMAIN_NODES];

        for i in 0..DOMAIN_NODES {
            let input = sram.read_domain_d(i as u8)?;
            let left = sram.read_domain_d(((i + DOMAIN_NODES - 1) % DOMAIN_NODES) as u8)?;
            let right = sram.read_domain_d(((i + 1) % DOMAIN_NODES) as u8)?;
            let output = (left + input + right) / 3.0;
            results[i] = QplResult { node: i, input, output };
        }

        // Modelo de latência: 2.37us + 0.01us por iteração adicional
        let extra_cycles = ((0.0 + (self.config.iterations as f64 * 0.01))
            * self.clock.freq_mhz as f64) as u64;
        self.clock.tick(extra_cycles);

        self.counters.qpl_cycles += self.clock.total_cycles() - start_cycles;
        self.config.status = 0x1; // Done
        self.config.control = 0;  // Auto-clear start

        Ok(results)
    }

    pub fn is_done(&self) -> bool {
        self.config.status & 0x1 != 0
    }

    pub fn counters(&self) -> &PerformanceCounters {
        &self.counters
    }
}

impl ArkhePeripheral for QplAccelerator {
    fn read_reg(&mut self, addr: u32) -> Result<u32, SocError> {
        match addr {
            0x00 => Ok(self.config.control),
            0x04 => Ok(self.config.status),
            0x08 => Ok(self.config.iterations),
            0x0C => Ok(self.config.threshold),
            _ => Err(SocError::InvalidAddress(addr)),
        }
    }

    fn write_reg(&mut self, addr: u32, val: u32) -> Result<(), SocError> {
        match addr {
            0x00 => {
                self.config.control = val;
                Ok(())
            }
            0x08 => {
                self.config.iterations = val;
                Ok(())
            }
            0x0C => {
                self.config.threshold = val;
                Ok(())
            }
            _ => Err(SocError::InvalidAddress(addr)),
        }
    }
}
