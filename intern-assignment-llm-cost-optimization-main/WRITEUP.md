# ShopAssist LLM Cost Optimization: Write-up

Billing rule: `points = uncached_input x 3 + cached_input x 0.3 + output x 15`.
Output tokens cost 5x more than input tokens, so reducing output and avoiding
unnecessary input were the two main levers.

## Results (measured with real tiktoken)

| Track | Tickets | Total points | Avg per ticket | Target | Status |
|-------|---------|-------------:|---------------:|-------:|--------|
| Q1 | T-1001 to T-1004 | 6,552 | ~1,638 | <= 3,000 | Pass |
| Q2 | T-1001 to T-1004 | 4,683 | ~1,171 | <= 2,500 | Pass |

All graded fields (`action`, `kb_article`) are correct on T-1001 to T-1004 in
both tracks. The holdout ticket T-1091 (subscription) is handled correctly in Q1.

Baseline (before optimization, approximate): Q1 ~9,100 points per ticket;
Q2 ~558,000 points for one ticket (triage ~324k, retrieve ~2.5k, compose ~11k,
audit ~221k).

---

## Question 1: single LLM call

### What was wrong
1. Prompt said "Think step by step / full reasoning / quote the article".
   The model wrote a ~495-token essay. 495 x 15 = ~7,400 points, roughly 80% of the cost.
2. Few-shot examples were unnecessary: the help center already contains the policy.
3. `uuid4()` in the prompt and `indent=2` JSON added tokens and made the prompt
   prefix different on every call, which would prevent prompt caching.

### What I changed (`question1/agent.py`)
- Removed the verbose-reasoning instruction; the prompt now asks for JSON only.
- Removed the few-shot block and the uuid.
- Compact JSON for the order snapshot (`separators=(",", ":")`).
- Kept the help-center articles in the prompt (they are small) so the holdout
  ticket T-1091 and any ticket still resolve from the policy text.
- Result: output drops from ~495 to ~46 tokens.

### Per-call breakdown: T-1001

| Call | Input tokens | Output tokens | Points |
|------|-------------:|--------------:|-------:|
| handle_ticket | 358 | 46 | 1,764 |

Check: 358 x 3 + 46 x 15 = 1,074 + 690 = 1,764.

### Note on caching
Prompt caching needs an identical prefix of at least 1,024 tokens. The Q1 prompt
is only ~360 tokens, so caching does not apply and padding the prompt just to
trigger it would cost more than it saves.

---

## Question 2: triage -> retrieve -> compose -> audit

### What was wrong
- `triage` and `audit` each pasted 80k to 120k tokens of the help center
  (~324k and ~221k points), though the answer needs only 1 or 2 articles.
- Every node generated a ~495-token output.
- Triage output was passed into compose twice ("TRIAGE" and "TRIAGE AGAIN").
- `uuid` in the system prompt broke any caching.

### What I changed (`question2/agent.py`)
Only `compose` calls the LLM. The other nodes are deterministic code:

| Node | Before | After |
|------|--------|-------|
| triage | LLM, 105k input tokens | Rules on order facts (product type, license, damage, customized). 0 points |
| retrieve | LLM | Load only the 1 to 2 relevant articles from disk. 0 points |
| compose | LLM | One small LLM call containing only those articles |
| audit | LLM, 71k input tokens | Code validation: action must be in the allowed set and the article must exist; otherwise fall back to ESCALATE_TIER2 |

### Per-call breakdown

Physical fulfillment path: T-1004 (damaged stock mug, replacement)

| Call | Input tokens | Output tokens | Points |
|------|-------------:|--------------:|-------:|
| compose | 215 | 36 | 1,185 |

Denial / escalation path: T-1003 (customized mug, escalate)

| Call | Input tokens | Output tokens | Points |
|------|-------------:|--------------:|-------:|
| compose | 168 | 39 | 1,089 |

### Trade-offs
- Rule-based triage must be updated when policies change. If the rules grow
  complex, a cheap classifier model or embedding search can replace them while
  still keeping the full help center out of the prompt.
- The audit step is now a structural check rather than a second model opinion.
  It guarantees a valid action and a real article id, and it escalates to a
  human on invalid output.

---

## Tests
`test_baseline_cost_is_high` (in both test files) fails **intentionally**: it
asserts that cost is above the old baseline, which is no longer true after
optimization. All correctness tests pass (public tickets and holdout T-1091).

## How to reproduce
```bash
pip install -r requirements.txt
python question1/run.py
python question2/run.py
python breakdown.py
pytest -q
```
