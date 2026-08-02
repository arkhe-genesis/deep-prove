use safety_core::seam_integrity::{ConsistencyResult, FactualEquivalence, SeamIntegrityMonitor, SemanticEquivalence};
use safety_core::veto::{AnubisVetoV3, RealMetrics, VetoAction};

struct MockPoint {
    semantic_id: usize,
    factual_id: usize,
}

impl SemanticEquivalence for MockPoint {
    fn semantic_eq(&self, other: &Self) -> bool {
        self.semantic_id == other.semantic_id
    }
}

impl FactualEquivalence for MockPoint {
    fn factual_eq(&self, other: &Self) -> bool {
        self.factual_id == other.factual_id
    }
}

#[test]
fn test_seam_integrity_monitor() {
    let monitor = SeamIntegrityMonitor::<MockPoint>::new(0.5);

    let p1 = MockPoint { semantic_id: 1, factual_id: 1 };
    let p2 = MockPoint { semantic_id: 1, factual_id: 1 }; // Consistent
    let p3 = MockPoint { semantic_id: 1, factual_id: 2 }; // HallucinationRisk
    let p4 = MockPoint { semantic_id: 2, factual_id: 1 }; // Paraphrase
    let p5 = MockPoint { semantic_id: 2, factual_id: 2 }; // Inconsistent

    assert_eq!(monitor.check(&p1, &p2), ConsistencyResult::Consistent);
    assert_eq!(monitor.check(&p1, &p3), ConsistencyResult::HallucinationRisk);
    assert_eq!(monitor.check(&p1, &p4), ConsistencyResult::Paraphrase);
    assert_eq!(monitor.check(&p1, &p5), ConsistencyResult::Inconsistent);
}

#[test]
fn test_anubis_veto() {
    let veto = AnubisVetoV3 {
        seam_tolerance: 0.5,
        entropy_limit: 1.0,
    };

    let metrics_ok = RealMetrics {
        perplexity: 1.0,
        token_entropy: 0.5,
        rag_density: 0.8,
    };
    let metrics_high_entropy = RealMetrics {
        perplexity: 1.0,
        token_entropy: 1.5,
        rag_density: 0.8,
    };

    // Test EntropySpike
    if let VetoAction::HaltAndLog(msg) = veto.evaluate(&ConsistencyResult::Consistent, &metrics_high_entropy, "test") {
        assert!(msg.contains("EntropySpike"));
    } else {
        panic!("Expected EntropySpike");
    }

    // Test HallucinationRisk
    if let VetoAction::HaltAndLog(msg) = veto.evaluate(&ConsistencyResult::HallucinationRisk, &metrics_ok, "test") {
        assert!(msg.contains("HallucinationRisk"));
    } else {
        panic!("Expected HallucinationRisk");
    }

    // Test Paraphrase (Allow)
    assert_eq!(veto.evaluate(&ConsistencyResult::Paraphrase, &metrics_ok, "test"), VetoAction::Allow);

    // Test Consistent (Allow)
    assert_eq!(veto.evaluate(&ConsistencyResult::Consistent, &metrics_ok, "test"), VetoAction::Allow);
}
