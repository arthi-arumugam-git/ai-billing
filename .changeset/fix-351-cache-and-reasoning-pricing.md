---
"@ai-billing/openai": patch
"@ai-billing/google": patch
---

fix: correct cached input and reasoning token pricing

Fixes three billing discrepancies reported in narevai/ai-billing#351:

1. **OpenAI reasoning tokens billed at $0**: When no `internalReasoningTokens` rate is set in the pricing table, reasoning tokens now correctly fall back to the `completionTokens` rate, matching the documented behavior in `CostInputs`.

2. **OpenAI cached input double-charged**: The middleware was passing `inputTokens.total` (which includes cached tokens) as `promptTokens`, then also adding `cacheReadTokens`. Now uses `inputTokens.noCache` when available, or computes `total - cacheRead` as fallback.

3. **Google cached input double-charged**: Same issue as OpenAI. The middleware was passing `promptTokenCount` (which includes `cachedContentTokenCount`) as `promptTokens`. Now correctly subtracts cached tokens.

Also applies the reasoning rate fallback fix to the Google cost calculator for consistency with the documented `CostInputs` semantics.

Both v3 and v4 middleware versions are fixed for each provider.
