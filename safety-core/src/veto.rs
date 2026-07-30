use crate::seam_integrity::ConsistencyResult;

/// Métricas reais do modelo. NÃO órfãs mais. Consumidas diretamente pelo Veto.
#[derive(Debug, Clone)]
pub struct RealMetrics {
    pub perplexity: f64,
    pub token_entropy: f64,
    pub rag_density: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VetoAction {
    Allow,
    HaltAndLog(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VetoReason {
    SeamRupture,
    LackOfEvidence,
    EntropySpike, // AGORA ESTÁ ATIVO
}

pub struct AnubisVetoV3 {
    pub seam_tolerance: f64,
    pub entropy_limit: f64,
}

impl AnubisVetoV3 {
    pub fn evaluate(
        &self,
        consistency: &ConsistencyResult,
        metrics: &RealMetrics, // AGORA É CONSUMIDO
        context: &str,
    ) -> VetoAction {
        // 1. Verificar a entropia PRIMEIRO (métrica de menor latência)
        if metrics.token_entropy > self.entropy_limit {
            return VetoAction::HaltAndLog(
                format!("[VETO] EntropySpike: {} > {}. Context: {}",
                    metrics.token_entropy, self.entropy_limit, context)
            );
        }

        // 2. Verificar a integridade da costura
        match consistency {
            ConsistencyResult::HallucinationRisk => {
                VetoAction::HaltAndLog(format!("[VETO] HallucinationRisk em '{}'", context))
            }
            ConsistencyResult::Paraphrase => {
                VetoAction::Allow // Falso negativo benigno
            }
            _ => VetoAction::Allow
        }
    }
}
