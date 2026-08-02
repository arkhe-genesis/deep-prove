use std::ffi::CStr;
use std::os::raw::c_char;
use serde::{Deserialize, Serialize};

use crate::seam_integrity::{ConsistencyResult, FactualEquivalence, SeamIntegrityMonitor, SemanticEquivalence};

#[derive(Serialize, Deserialize)]
pub struct LeanVerifiablePoint {
    pub point_data: String,
    pub evidence_hash: String,
    pub semantic_signature: Vec<f32>,  // embedding para semantic_eq
}

impl SemanticEquivalence for LeanVerifiablePoint {
    fn semantic_eq(&self, other: &Self) -> bool {
        if self.semantic_signature.len() != other.semantic_signature.len() {
            return false;
        }
        if self.semantic_signature.is_empty() {
            return false; // Evita divisão por zero
        }
        let dot: f32 = self.semantic_signature.iter()
            .zip(&other.semantic_signature)
            .map(|(a, b)| a * b)
            .sum();
        let norm_a = self.semantic_signature.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = other.semantic_signature.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { return false; }
        let cosine = dot / (norm_a * norm_b);
        cosine > 0.85  // threshold configurável
    }
}

impl FactualEquivalence for LeanVerifiablePoint {
    fn factual_eq(&self, other: &Self) -> bool {
        // Factual equivalence = mesma evidence OU evidências mutuamente corroborantes
        self.evidence_hash == other.evidence_hash
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn arkhe_rust_check_seam_integrity(
    a_json_ptr: *const c_char,
    b_json_ptr: *const c_char,
) -> bool {
    // --- Segurança: validação de entrada ---
    if a_json_ptr.is_null() || b_json_ptr.is_null() {
        return false;
    }

    // Limite de 64KB para prevenir DoS
    const MAX_LEN: usize = 65536;

    let a_str = unsafe {
        let bytes = CStr::from_ptr(a_json_ptr).to_bytes();
        if bytes.len() > MAX_LEN { return false; }
        match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let b_str = unsafe {
        let bytes = CStr::from_ptr(b_json_ptr).to_bytes();
        if bytes.len() > MAX_LEN { return false; }
        match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let a: LeanVerifiablePoint = match serde_json::from_str(a_str) {
        Ok(val) => val,
        Err(_) => return false,
    };
    let b: LeanVerifiablePoint = match serde_json::from_str(b_str) {
        Ok(val) => val,
        Err(_) => return false,
    };

    // --- Verificação via SeamIntegrityMonitor ---
    let monitor = SeamIntegrityMonitor::<LeanVerifiablePoint>::new(0.5); // Valor dummy para entropy
    let result = monitor.check(&a, &b);

    // checkFaithful (check_seam_integrity) retorna true APENAS quando semântico e factual batem (Consistent)
    // ou quando factual bate mas semântico não (Paraphrase — falso negativo aceitável).
    // Retorna false quando semântico bate mas factual não (HallucinationRisk).
    matches!(result, ConsistencyResult::Consistent | ConsistencyResult::Paraphrase)
}

#[unsafe(no_mangle)]
pub extern "C" fn arkhe_rust_get_entropy(logits_json_ptr: *const c_char) -> f64 {
    if logits_json_ptr.is_null() {
        return 0.0;
    }

    // Limite de 64KB para prevenir DoS
    const MAX_LEN: usize = 65536;

    let _logits_str = unsafe {
        let bytes = CStr::from_ptr(logits_json_ptr).to_bytes();
        if bytes.len() > MAX_LEN { return 0.0; }
        match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return 0.0,
        }
    };

    // In a real implementation, we would parse the JSON and calculate entropy
    // Here we just return a dummy value
    0.5
}
