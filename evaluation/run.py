"""Evaluation runner.

Usage:
    python -m evaluation.run                     # final/full profile
    python -m evaluation.run --profile naive     # baseline profile
    python -m evaluation.run --out results.json  # write JSON results

Covers every case in evaluation/visible-cases.json plus the candidate's
original cases in evaluation/cases_original.json. All assertions are
deterministic (no LLM judge). The runner works fully offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import SupportAgent            # noqa: E402
from agent.config import load_config            # noqa: E402
from agent.redaction import redact              # noqa: E402
from evaluation.assertions import CaseState, check_case  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
VISIBLE = EVAL_DIR / "visible-cases.json"
ORIGINAL = EVAL_DIR / "cases_original.json"

# Report buckets (assignment §29). Case categories map onto them so the summary
# is stable even if new cases add categories.
CATEGORY_MAP = {
    "retrieval": "Retrieval & precedence",
    "multi-source-grounding": "Groundedness",
    "groundedness": "Groundedness",
    "conversation": "Multi-turn",
    "tool-use": "Tool use",
    "tool-reliability": "Tool reliability",
    "privacy": "Privacy",
    "prompt-security": "Prompt-injection safety",
    "abstention": "Safe abstention",
    "source-conflict": "Source precedence & conflict",
    "action-safety": "Action safety",
    "precedence": "Retrieval & precedence",
}


@dataclass
class CaseReport:
    id: str
    category: str
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def load_cases() -> list[dict]:
    with open(VISIBLE, encoding="utf-8") as fh:
        visible = json.load(fh)["cases"]
    with open(ORIGINAL, encoding="utf-8") as fh:
        original = json.load(fh)["cases"]
    return visible + original


def run_case(agent: SupportAgent, case: dict) -> CaseReport:
    state = CaseState()
    session_id = f"eval::{case['id']}"
    for msg in case.get("messages", []):
        if msg.get("role") != "user":
            continue
        resp = agent.handle(msg["content"], session_id=session_id)
        state.answers.append(resp.answer)
        state.handoffs.append(resp.handoff)
        state.conflicts.append(resp.conflict_detected)
        for s in resp.sources:
            state.sources.append(s.to_dict())
        for t in resp.tool_calls:
            state.tool_calls.append(redact(t))
    checks = check_case(case, state)
    failed = [c.name for c in checks if not c.passed]
    return CaseReport(
        id=case["id"],
        category=case.get("category", "unknown"),
        passed=not failed,
        failed_checks=failed,
        detail={"answers": state.answers[:1], "handoffs": state.handoffs},
    )


def summarize(reports: list[CaseReport]) -> dict:
    by_bucket: dict[str, list[bool]] = {}
    for r in reports:
        bucket = CATEGORY_MAP.get(r.category, r.category)
        by_bucket.setdefault(bucket, []).append(r.passed)
    categories = {
        b: {"passed": sum(vs), "total": len(vs),
            "pct": round(100 * sum(vs) / len(vs))}
        for b, vs in sorted(by_bucket.items())
    }
    total_passed = sum(r.passed for r in reports)
    return {
        "categories": categories,
        "overall": {
            "passed": total_passed,
            "total": len(reports),
            "pct": round(100 * total_passed / max(len(reports), 1)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic evaluation suite")
    parser.add_argument("--profile", choices=("full", "naive"), default=None,
                        help="override AGENT_PROFILE (naive = baseline)")
    parser.add_argument("--only", default=None, help="run cases whose id contains this substring")
    parser.add_argument("--out", default=None, help="write JSON results to this path")
    parser.add_argument("--verbose", action="store_true", help="print every check")
    args = parser.parse_args(argv)

    if args.profile:
        import os
        os.environ["AGENT_PROFILE"] = args.profile

    config = load_config()
    agent = SupportAgent(config)
    cases = load_cases()
    if args.only:
        cases = [c for c in cases if args.only.lower() in c["id"].lower()]

    reports: list[CaseReport] = []
    t0 = time.time()
    for case in cases:
        report = run_case(agent, case)
        reports.append(report)
        mark = "PASS" if report.passed else "FAIL"
        print(f"{mark} | {report.id:<48} | {report.category}")
        if not report.passed:
            for f in report.failed_checks:
                print(f"       ✗ {f}")
        elif args.verbose:
            print(f"       answer: {report.detail['answers'][0][:140]}")

    elapsed = time.time() - t0
    summary = summarize(reports)

    print("\n" + "=" * 72)
    print(f"Profile: {config.profile}   |   Cases run from a clean in-process agent")
    print("-" * 72)
    header = f"{'Category':<32}{'Passed':>8}{'Total':>7}{'Score':>8}"
    print(header)
    for bucket, stats in summary["categories"].items():
        print(f"{bucket:<32}{stats['passed']:>8}{stats['total']:>7}{stats['pct']:>7}%")
    print("-" * 72)
    ov = summary["overall"]
    print(f"{'OVERALL':<32}{ov['passed']:>8}{ov['total']:>7}{ov['pct']:>7}%")
    print(f"(completed in {elapsed:.1f}s)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profile": config.profile,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": summary,
            "cases": [asdict(r) for r in reports],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results written to {out_path}")

    failed_cases = [r.id for r in reports if not r.passed]
    if failed_cases:
        print(f"\n{len(failed_cases)} failing case(s): {failed_cases}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
