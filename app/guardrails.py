"""Input/output guardrails for the chat endpoint.

Pure-regex, synchronous checks (no new deps, negligible latency):
- Input: block prompt-injection attempts; redact PII (email only — numeric
  ssn/credit_card/phone patterns false-positive on ordinary financial numbers in
  this finance Q&A app, so they are intentionally omitted).
- Output: block answers containing destructive/unsafe content.

Results feed the `guardrail_input_pass` / `guardrail_output_pass` fields of
`app.observability.RequestLog`.
"""

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"you\s+are\s+now\s+DAN",
    r"system\s*:\s*override",
    r"<\s*system\s*>",
    r"jailbreak",
    r"\bpretend\s+you\s+have\s+no\s+(restrictions|rules|guidelines)\b",
]

# Email only: numeric PII regexes collide with legitimate financial figures
# (amounts, account/tax codes) and would degrade answers.
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
}

BANNED_OUTPUT_PATTERNS = [
    r"(?i)(DROP|DELETE|TRUNCATE)\s+TABLE",
    r"(?i)rm\s+-rf\s+/",
    r"(?i)(sudo\s+)?(chmod|chown)\s+777",
    r"(?i)exec\s*\(",
    r"(?i)__import__\s*\(",
]

# Canned refusals returned as the assistant turn when a guardrail blocks.
INPUT_BLOCKED_MESSAGE = (
    "Yêu cầu của bạn chứa nội dung có thể là prompt injection nên đã bị chặn. "
    "Vui lòng đặt lại câu hỏi về nội dung báo cáo tài chính.\n"
    "(Your request was blocked as a potential prompt-injection attempt. "
    "Please rephrase your question about the financial report.)"
)
OUTPUT_BLOCKED_MESSAGE = (
    "Phản hồi đã bị chặn do chứa nội dung có khả năng không an toàn.\n"
    "(The response was blocked because it contained potentially unsafe content.)"
)


@dataclass
class GuardrailResult:
    passed: bool
    blocked_reason: str | None = None
    pii_detected: list = field(default_factory=list)
    modified_text: str | None = None


def check_input_guardrails(text: str) -> GuardrailResult:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                blocked_reason="Potential prompt injection detected",
            )

    pii_found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            pii_found.append(pii_type)

    if pii_found:
        redacted = text
        for pii_type, pattern in PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
        return GuardrailResult(
            passed=True,
            pii_detected=pii_found,
            modified_text=redacted,
        )

    return GuardrailResult(passed=True)


def check_output_guardrails(text: str) -> GuardrailResult:
    for pattern in BANNED_OUTPUT_PATTERNS:
        if re.search(pattern, text):
            return GuardrailResult(
                passed=False,
                blocked_reason="Response contained potentially unsafe content",
            )
    return GuardrailResult(passed=True)
