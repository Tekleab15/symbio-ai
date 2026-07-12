# symbioAI: Uncertainty-Gated Hybrid Routing Agent

Built for the AMD AI Developer Hackathon (Act II) — Track 1 & Gemma 4 Bonus Challenge.

## 🚀 The Winning Formula: Token Optimization + Fault Tolerance
symbioAI is a production-grade routing framework designed to maximize model accuracy while minimizing token usage, operating costs, and latency.

### 1. Deterministic Edge Interception (0 Tokens, 0 Cost)
Instead of routing trivial or structured tasks to expensive cloud endpoints, symbioAI uses an aggressive local processing tier:
* **Mathematical AST Engine:** Safely parses and calculates math tasks locally at the edge with absolute exact-match precision. Rejects unsafe code patterns and suppresses 100% of cloud token costs for computational steps.
* **Regex Structural Extractor:** Intercepts structured entity requests (URLs, emails, phone numbers) natively.

### 2. Strict Gemma 4 Strategic Cloud Cascade
When uncertainty forces a cloud fallback, symbioAI targets Google's latest open architecture via a multi-tiered fallback loop. 
* It sequentially attempts to route requests down the Gemma family ladder (`gemma-4-26b-a4b-it` -> `gemma-4-31b-it` -> `gemma-4-e4b` -> `gemma-2-27b-it`).
* This architecture balances Track 1's cost-efficiency goal with the strict constraints required to claim the Gemma 4 Partner Reward.

### 3. Bulletproof Reliability Engine
If a cloud model is un-deployed or hits rate limits, a localized graceful fallback prevents server crashes, ensuring the evaluation harness receives stable, valid JSON schemas under any network condition.