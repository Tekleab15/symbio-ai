# symbioAI: Uncertainty-Gated Hybrid Routing Agent

Built for the AMD AI Developer Hackathon (Act II) — Track 1 & Gemma 4 Bonus Challenge.

## 🚀 The Winning Formula: Token Optimization + Fault Tolerance
symbioAI is a production-grade routing framework designed to maximize model accuracy while minimizing token usage, operating costs, and latency.

### 1. Deterministic Edge Interception (0 Tokens, 0 Cost)
Instead of routing trivial or structured tasks to expensive cloud endpoints, symbioAI uses an aggressive local processing tier:
* **Mathematical AST Engine:** Safely parses and calculates math tasks locally at the edge with absolute exact-match precision. Rejects unsafe code patterns and suppresses 100% of cloud token costs for computational steps.
* **Regex Structural Extractor:** Intercepts structured entity requests (URLs, emails, phone numbers) natively.

## 💎 Best Use of Gemma 4 – Enterprise Query Router
SymbioAI incorporates a production-grade **Gemma 4 Cascade Engine** specifically architected to absorb on-demand container lifecycle latencies. 

### Dynamic Failover Loop & 503 Warming retry Logic
* **Gemma Prioritization Architecture:** When the environmental toggle `SYMBIO_GEMMA_FIRST=1` is flipped, our router shifts workloads away from serverless endpoints, funneling reasoning tasks directly to private On-Demand deployments (`accounts/YOUR_ACCOUNT/deployments/YOUR_DEPLOYMENT`).

* **Hardware Latency Absorption:** Because on-demand models scale to zero to minimize resource expenditures, our backend is equipped with an asynchronous backoff loop (`0s -> 8s -> 20s -> 40s`). It intercepts `503 Service Unavailable / Scaling from Zero` server anomalies gracefully, stabilizing the engine until the Gemma context layer is warm.
* **Resilient Graceful Degradation:** If on-demand quotas are exhausted, our engine cleanly shifts downstream tasks to public model instances, preventing runtime failures.

### 3. Bulletproof Reliability Engine
If a cloud model is un-deployed or hits rate limits, a localized graceful fallback prevents server crashes, ensuring the evaluation harness receives stable, valid JSON schemas under any network condition.

## Judge Notes

symbioAI is correctness-first. It solves mechanically verifiable tasks locally for zero counted Fireworks tokens, then escalates unresolved tasks to official Fireworks models.

Default runtime routing:
- Math / safe extraction / obvious mixed sentiment: local deterministic solvers.
- General, factual QA, summarization, NER, logic: `minimax-m3`.
- Code generation and debugging: `kimi-k2p7-code`.
- Gemma 4 partner route: configurable via `FIREWORKS_MODEL_GEMMA`, `FIREWORKS_MODEL_FALLBACKS`, and `SYMBIO_GEMMA_FIRST=1`.

The container does not require local model weights, Ollama, vLLM, or GPU runtime. It is designed for edge grading environment.