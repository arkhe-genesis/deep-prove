#!/usr/bin/env python3
"""
claim_verifier.py

Verificador de afirmações (claims) em documentos técnicos.

PRINCÍPIO DE DESIGN: toda classificação deve vir de um teste mecânico,
reproduzível e inspecionável — nunca de um número de confiança atribuído
por "sensação" do modelo. Se um critério não pode ser calculado a partir
do texto de entrada, ele NÃO é usado, e isso é declarado explicitamente
nas limitações do relatório.

O que este script FAZ:
  - Extrai claims (sentenças/bullets declarativos) de um documento .md/.txt
  - Detecta presença de citação/fonte (regex sobre padrões de referência)
  - Detecta similaridade textual entre claims (ECHO real: repetição
    detectável, via difflib.SequenceMatcher — não "incompressibilidade
    algorítmica" de fachada)
  - Detecta conectores causais sem termo técnico de mecanismo próximo
  - Detecta jargão não definido no próprio documento (termos em
    maiúscula/símbolos que nunca são explicados)

O que este script NÃO FAZ (e por quê):
  - Não avalia verdade factual do conteúdo (isso exige verificação
    externa: busca, execução de código, prova formal — fora do escopo
    de um verificador puramente textual)
  - Não atribui probabilidades numéricas de confiança — os testes
    disponíveis não sustentam uma medida de probabilidade calibrada;
    fingir uma seria exatamente o "ECO apresentado como PRESENÇA" que
    o próprio documento auditado alega evitar
  - Não roda Lean4/Kani/TLA+/Z3 — nenhum desses é invocado; se um claim
    menciona esses verificadores sem de fato submetê-los a eles, isso é
    sinalizado como "verificador citado mas não executado"
"""

import re
import sys
import json
import argparse
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime, timezone


CITATION_PATTERNS = [
    r"https?://\S+",
    r"\[\d+\]",
    r"\bdoi:\S+",
    r"\bcf\.\s",
    r"\bver\s+(seção|apêndice|anexo)\s+[\w.]+",
    r"\(cite[d]?\b",
    r"\bsegundo\s+[A-ZÀ-Ú][\wÀ-ÿ]+",  # "segundo Fulano"
]

VERIFIER_NAME_PATTERN = re.compile(
    r"\b(Lean4|Lean\s?4|Kani|TLA\+|Z3|Coq|Isabelle)\b", re.IGNORECASE
)

CAUSAL_CONNECTORS = [
    "porque", "logo,", "portanto", "leva a", "causa", "causam",
    "resulta em", "implica", "faz com que", "gera", "produz",
]

MECHANISM_MARKERS = [
    "algoritmo", "protocolo", "função", "equação", "teorema",
    "hash", "assinatura", "compilador", "teste", "prova",
    "mecanismo", "invariante", "circuito", "processo formal",
]

# termos "teatrais": símbolos/jargão que soam formais mas raramente são
# definidos operacionalmente no corpo do documento
JARGON_CANDIDATES = re.compile(
    r"\b([A-ZÀ-Ú]{2,}(?:[- ][A-ZÀ-Ú]{2,})*|"
    r"τ\([^)]*\)|"
    r"\b[A-ZÀ-Ú][a-zà-ÿ]+(?:\s[A-ZÀ-Ú][a-zà-ÿ]+){1,3}(?=\s+(?:framework|protocol|protocolo)))\b"
)


@dataclass
class ClaimResult:
    idx: int
    text: str
    label: str
    tests: dict = field(default_factory=dict)
    rationale: str = ""


def split_claims(text: str) -> list[str]:
    """Extrai claims: bullets (•, -, *, números) e sentenças declarativas
    de parágrafos. Ignora cabeçalhos, separadores e linhas vazias."""
    lines = text.splitlines()
    claims = []
    buf = []

    def flush():
        if buf:
            para = " ".join(buf).strip()
            if para:
                for sent in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú])", para):
                    sent = sent.strip()
                    if len(sent.split()) >= 4:
                        claims.append(sent)
            buf.clear()

    for line in lines:
        raw = line.strip()
        if not raw or set(raw) <= set("═-*—_ "):
            flush()
            continue
        if re.match(r"^\[\d+\]|^#{1,6}\s|^[A-Z0-9\s\[\]()áéíóúâêôãõçÀ-Ú]{6,}$", raw) and len(raw.split()) <= 8:
            # provável cabeçalho/título de seção — não é claim
            flush()
            continue
        bullet_match = re.match(r"^[•\-\*×]\s*(.+)$|^[a-e]\.\s+(.+)$", raw)
        if bullet_match:
            flush()
            content = bullet_match.group(1) or bullet_match.group(2)
            if len(content.split()) >= 4:
                claims.append(content.strip())
            continue
        buf.append(raw)
    flush()
    return claims


def has_citation(claim: str) -> bool:
    return any(re.search(p, claim, re.IGNORECASE) for p in CITATION_PATTERNS)


def mentions_verifier_uninvoked(claim: str) -> bool:
    """True se cita um verificador formal por nome sem evidência de
    que o script/pipeline atual o executou (este verificador nunca
    executa nenhum, então qualquer menção é 'citado, não executado')."""
    return bool(VERIFIER_NAME_PATTERN.search(claim))


def causal_without_mechanism(claim: str) -> bool:
    low = claim.lower()
    has_causal = any(
        re.search(r'(?<!\w)' + re.escape(c) + r'(?!\w)', low)
        for c in CAUSAL_CONNECTORS
    )
    has_mechanism = any(
        re.search(r'(?<!\w)' + re.escape(m) + r'(?!\w)', low)
        for m in MECHANISM_MARKERS
    )
    return has_causal and not has_mechanism


def jargon_terms(claim: str) -> list[str]:
    found = set()
    for m in JARGON_CANDIDATES.finditer(claim):
        term = m.group(0).strip()
        if len(term) > 1 and term.upper() not in {"P", "ECHO", "LEMMA", "THEOREM", "CONJECTURE"}:
            found.add(term)
    return sorted(found)


def echo_pairs(claims: list[str], threshold: float = 0.72) -> dict[int, list[tuple[int, float]]]:
    """Retorna, para cada índice de claim, a lista de (índice do outro
    claim, razão de similaridade) acima do threshold — repetição
    detectável mecanicamente via SequenceMatcher."""
    result: dict[int, list[tuple[int, float]]] = {}
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            ratio = SequenceMatcher(None, claims[i].lower(), claims[j].lower()).ratio()
            if ratio >= threshold:
                result.setdefault(i, []).append((j, ratio))
                result.setdefault(j, []).append((i, ratio))
    return result


def defined_terms(text: str) -> set[str]:
    """Termos que o próprio documento define explicitamente
    (padrão 'TERMO — definição' ou 'TERMO: definição')."""
    defs = set()
    for m in re.finditer(r"([A-ZÀ-Ú][A-ZÀ-Ú \-]{2,})\s*[—:]\s+\S", text):
        defs.add(m.group(1).strip())
    return defs


def classify(claim: str, idx: int, all_claims: list[str],
             echo_map: dict[int, list[tuple[int, float]]],
             doc_defined_terms: set[str]) -> ClaimResult:
    tests = {}

    cited = has_citation(claim)
    tests["citação_externa_detectada"] = cited

    verifier_cited = mentions_verifier_uninvoked(claim)
    tests["cita_verificador_formal_sem_execução"] = verifier_cited

    causal_gap = causal_without_mechanism(claim)
    tests["conectivo_causal_sem_mecanismo_próximo"] = causal_gap

    jt = jargon_terms(claim)
    undefined_jargon = [t for t in jt if t not in doc_defined_terms]
    tests["jargão_não_definido_no_documento"] = undefined_jargon

    is_echo = idx in echo_map
    tests["similaridade_com_outro_claim"] = (
        [{"claim_idx": j, "ratio": round(r, 2)} for j, r in echo_map[idx]]
        if is_echo else []
    )

    # --- lógica de classificação (determinística, na ordem abaixo) ---
    if is_echo:
        label = "ECHO"
        rationale = (
            f"Similaridade textual ≥0.72 com claim(s) "
            f"{[j for j, _ in echo_map[idx]]} — repetição de padrão sem "
            f"novidade informacional detectável no texto."
        )
    elif verifier_cited and not cited:
        label = "CONJECTURE"
        rationale = (
            "Cita verificador formal por nome (Lean4/Kani/TLA+/Z3 etc.) mas "
            "nenhuma evidência de execução está presente no texto — "
            "invocação retórica, não prova. Rebaixado a CONJECTURE."
        )
    elif causal_gap:
        label = "CONJECTURE"
        rationale = (
            "Usa conectivo causal (porque/logo/implica/etc.) sem termo de "
            "mecanismo técnico nas proximidades — atribui causalidade sem "
            "mecanismo explicitado."
        )
    elif undefined_jargon:
        label = "CONJECTURE"
        rationale = (
            f"Introduz termo(s) não definidos no documento: {undefined_jargon}. "
            f"Sem definição operacional, o termo não é verificável "
            f"externamente por um leitor sem contexto interno."
        )
    elif cited:
        label = "LEMMA"
        rationale = (
            "Contém marcador de citação/referência, mas este verificador "
            "não confirma a fonte de fato (isso exigiria fetch/verificação "
            "externa, fora do escopo textual) — por isso LEMMA, não THEOREM."
        )
    else:
        label = "CONJECTURE"
        rationale = (
            "Nenhum dos testes disponíveis (citação, verificador executado, "
            "mecanismo causal, definição de jargão) foi satisfeito o "
            "suficiente para elevar a classificação — default para "
            "CONJECTURE por ausência de evidência mecânica, não por "
            "julgamento de plausibilidade."
        )

    return ClaimResult(idx=idx, text=claim, label=label, tests=tests, rationale=rationale)


def run(path: Path, threshold: float) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    claims = split_claims(text)
    echo_map = echo_pairs(claims, threshold=threshold)
    doc_defs = defined_terms(text)

    results = [
        classify(c, i, claims, echo_map, doc_defs) for i, c in enumerate(claims)
    ]

    counts = {"THEOREM": 0, "LEMMA": 0, "CONJECTURE": 0, "ECHO": 0}
    for r in results:
        counts[r.label] += 1

    return {
        "source_file": str(path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_claims": len(claims),
        "label_counts": counts,
        "echo_similarity_threshold": threshold,
        "results": [
            {
                "idx": r.idx,
                "claim": r.text,
                "label": r.label,
                "tests": r.tests,
                "rationale": r.rationale,
            }
            for r in results
        ],
        "limitations": [
            "Não avalia verdade factual — apenas estrutura textual e presença/ausência de marcadores mecânicos.",
            "Não atribui probabilidades numéricas de confiança: os testes disponíveis não sustentam uma medida calibrada.",
            "Não executa nenhum verificador formal (Lean4/Kani/TLA+/Z3); menções a eles são sinalizadas como não-executadas, nunca como prova.",
            "Detecção de jargão e de mecanismo causal é heurística baseada em listas de palavras — falsos positivos/negativos são esperados e devem ser revisados por humano.",
            "THEOREM nunca é atribuído automaticamente por este script: exigiria confirmação externa (execução de teste, fetch de fonte, checagem de prova) que está fora do escopo puramente textual.",
        ],
    }


def to_markdown(report: dict) -> str:
    lines = []
    lines.append(f"# Relatório de auditoria de claims — `{Path(report['source_file']).name}`")
    lines.append(f"\nGerado em: {report['generated_at_utc']}")
    lines.append(f"\nClaims extraídos: **{report['n_claims']}**")
    lines.append("\n## Distribuição")
    for label in ["THEOREM", "LEMMA", "CONJECTURE", "ECHO"]:
        lines.append(f"- **{label}**: {report['label_counts'][label]}")

    lines.append("\n## Claims classificados\n")
    for r in report["results"]:
        lines.append(f"### [{r['idx']}] `{r['label']}`")
        lines.append(f"> {r['claim']}")
        lines.append(f"\n**Razão:** {r['rationale']}")
        interesting_tests = {k: v for k, v in r["tests"].items() if v}
        if interesting_tests:
            lines.append(f"\n**Testes que dispararam:** `{json.dumps(interesting_tests, ensure_ascii=False)}`")
        lines.append("")

    lines.append("## Limitações declaradas do verificador\n")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Verificador mecânico de claims (THEOREM/LEMMA/CONJECTURE/ECHO)")
    ap.add_argument("path", type=Path, help="Arquivo .md/.txt a auditar")
    ap.add_argument("--threshold", type=float, default=0.72, help="Limiar de similaridade para ECHO (default 0.72)")
    ap.add_argument("--json-out", type=Path, default=None, help="Caminho para salvar relatório JSON")
    ap.add_argument("--md-out", type=Path, default=None, help="Caminho para salvar relatório Markdown")
    args = ap.parse_args()

    report = run(args.path, args.threshold)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.md_out:
        args.md_out.write_text(to_markdown(report), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(to_markdown(report))


if __name__ == "__main__":
    main()
