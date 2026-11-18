import { generateText, streamText, wrapLanguageModel } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createGoogleV4Middleware } from './language-model-v4-google-billing-middleware.js';
import {
  BillingEventSchema,
  MockLanguageModelV4,
  convertArrayToReadableStream,
} from '@ai-billing/testing';
import { LanguageModelV4GenerateResult } from '@ai-sdk/provider';
import type { BillingEvent, ModelPricing } from '@ai-billing/types';
import { z } from 'zod';

describe('GoogleBillingMiddlewareV4 Integration', () => {
  const StrictBillingEventSchema: z.ZodType<BillingEvent> = BillingEventSchema;
  const mockPricing: ModelPricing = {
    promptTokens: 0.000002, // $2.00 per 1M
    completionTokens: 0.000012, // $12.00 per 1M
    inputCacheReadTokens: 0.0000002, // $0.20 per 1M
    inputCacheWriteTokens: 0.000000375, // $0.375 per 1M
    internalReasoningTokens: 0.000012, // $12.00 per 1M
    request: 0,
  };

  const mockPriceResolver = vi.fn().mockResolvedValue(mockPricing);

  const createResult = (
    overrides: Partial<LanguageModelV4GenerateResult> = {},
  ): LanguageModelV4GenerateResult => ({
    content: [{ type: 'text', text: 'Paris' }],
    warnings: [],
    finishReason: { unified: 'stop', raw: 'stop' },
    usage: {
      inputTokens: { total: 16, noCache: 16, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 396, text: 54, reasoning: 342 }, // 54 text + 342 reasoning = 396
    },
    response: { id: 'resp_0e07e75c3318ef', timestamp: new Date() },
    providerMetadata: {
      google: {
        usageMetadata: {
          promptTokenCount: 16,
          candidatesTokenCount: 54,
          thoughtsTokenCount: 342,
          cachedContentTokenCount: 0,
        },
      },
    },
    ...overrides,
  });

  describe('wrapGenerate', () => {
    it('should extract usage, resolve pricing, calculate cost, and broadcast event', async () => {
      const destinationSpy = vi.fn();
      const middleware = createGoogleV4Middleware({
        destinations: [destinationSpy],
        priceResolver: mockPriceResolver,
      });

      const baseResult = createResult();
      const mockModel = new MockLanguageModelV4({
        modelId: 'gemini-3.1-pro-preview-customtools',
        provider: 'google',
        doGenerate: async () => baseResult,
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });

      await generateText({ model: wrappedModel, prompt: 'Capital of France?' });

      expect(mockPriceResolver).toHaveBeenCalledWith({
        modelId: 'gemini-3.1-pro-preview-customtools',
        providerId: 'google',
      });

      // Calculate expected cost:
      // Prompt: 16 * 0.000002 * 1e9 = 32,000 nanos
      // Completion: 396 * 0.000012 * 1e9 = 4,752,000 nanos
      // Total: 4,784,000 nanos ($0.004784)
      const expectedEvent = StrictBillingEventSchema.parse({
        generationId: baseResult.response?.id,
        modelId: mockModel.modelId,
        provider: mockModel.provider,
        usage: {
          inputTokens: 16,
          outputTokens: 396,
          cacheReadTokens: 0,
          reasoningTokens: 342,
        },
        cost: {
          amount: 4784000,
          unit: 'nanos',
          currency: 'USD',
        },

        tags: {},
      });

      expect(destinationSpy).toHaveBeenCalledTimes(1);
      const emittedPayload = destinationSpy.mock.calls[0]![0];
      let parsedEmittedEvent: BillingEvent;

      expect(() => {
        parsedEmittedEvent = StrictBillingEventSchema.parse(emittedPayload);
      }).not.toThrow();

      expect(parsedEmittedEvent!).toMatchObject(expectedEvent!);
    });
  });

  describe('wrapStream', () => {
    it('should extract usage and calculate cost correctly from stream finish chunk', async () => {
      const destinationSpy = vi.fn();
      const middleware = createGoogleV4Middleware({
        destinations: [destinationSpy],
        priceResolver: mockPriceResolver,
      });

      const baseResult = createResult();

      const mockModel = new MockLanguageModelV4({
        modelId: 'gemini-3.1-pro-preview-customtools',
        provider: 'google',
        doStream: async () => ({
          stream: convertArrayToReadableStream([
            { type: 'response-metadata', id: baseResult.response!.id! },
            { type: 'text-start', id: baseResult.response!.id! },
            {
              type: 'finish',
              finishReason: baseResult.finishReason,
              usage: baseResult.usage,
              providerMetadata: baseResult.providerMetadata,
            },
          ]),
          response: baseResult.response,
        }),
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
      const result = streamText({ model: wrappedModel, prompt: 'Hello' });
      await result.text;

      // 1. Wait for the middleware to broadcast
      await vi.waitFor(
        () => {
          expect(destinationSpy).toHaveBeenCalledTimes(1);
        },
        { timeout: 500 },
      );

      // 2. Define expected valid data
      const expectedEvent = StrictBillingEventSchema.parse({
        generationId: baseResult.response?.id,
        modelId: mockModel.modelId,
        provider: mockModel.provider,
        usage: {
          inputTokens: 16,
          outputTokens: 396,
          cacheReadTokens: 0,
          reasoningTokens: 342,
        },
        cost: {
          amount: 4784000,
          unit: 'nanos',
          currency: 'USD',
        },
        tags: {},
      });

      // 3. Extract, validate, and compare actual data
      const emittedPayload = destinationSpy.mock.calls[0]![0];
      let parsedEmittedEvent: BillingEvent;

      expect(() => {
        parsedEmittedEvent = StrictBillingEventSchema.parse(emittedPayload);
      }).not.toThrow();

      expect(parsedEmittedEvent!).toMatchObject(expectedEvent);
    });
  });

  it('should omit the cost object entirely if pricing resolves to undefined', async () => {
    const destinationSpy = vi.fn();
    const missingPriceResolver = vi.fn().mockResolvedValue(undefined);

    const middleware = createGoogleV4Middleware({
      destinations: [destinationSpy],
      priceResolver: missingPriceResolver,
    });

    const baseResult = createResult();
    const mockModel = new MockLanguageModelV4({
      modelId: 'unknown-future-gemini-model',
      doGenerate: async () => baseResult,
    });

    const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });

    await generateText({ model: wrappedModel, prompt: 'Hello' });

    const expectedEvent = StrictBillingEventSchema.parse({
      generationId: baseResult.response?.id,
      modelId: mockModel.modelId,
      provider: 'google',
      usage: {
        inputTokens: 16,
        outputTokens: 396,
        cacheReadTokens: 0,
        reasoningTokens: 342,
      },
      tags: {},
    });

    expect(destinationSpy).toHaveBeenCalledTimes(1);
    const emittedPayload = destinationSpy.mock.calls[0]![0];
    let parsedEmittedEvent: BillingEvent;

    expect(() => {
      parsedEmittedEvent = StrictBillingEventSchema.parse(emittedPayload);
    }).not.toThrow();

    expect(parsedEmittedEvent!).toMatchObject(expectedEvent);
    expect(parsedEmittedEvent!).not.toHaveProperty('cost');
  });

  it('should hit all fallback branches for full coverage (UUID generation, missing metadata)', async () => {
    const destinationSpy = vi.fn();
    const middleware = createGoogleV4Middleware({
      destinations: [destinationSpy],
      priceResolver: mockPriceResolver,
    });

    const baseResult = createResult({
      response: { id: undefined }, // Forces UUID generation
      providerMetadata: {}, // Forces fallback to 0 for all usageMetadata fields
    });

    const mockModel = new MockLanguageModelV4({
      modelId: 'gemini-1.5-flash',
      provider: '', // Forces provider fallback
      doGenerate: async () => baseResult,
    });

    const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
    await generateText({ model: wrappedModel, prompt: 'Hi' });

    await vi.waitFor(() => expect(destinationSpy).toHaveBeenCalledTimes(1));
    const emittedPayload = destinationSpy.mock.calls[0]![0];
    let parsedEmittedEvent: BillingEvent;

    expect(() => {
      parsedEmittedEvent = StrictBillingEventSchema.parse(emittedPayload);
    }).not.toThrow();

    const expectedEvent = StrictBillingEventSchema.parse({
      generationId: parsedEmittedEvent!.generationId, // Inject the random UUID
      modelId: mockModel.modelId,
      provider: 'google', // Expected fallback
      usage: {
        inputTokens: 0,
        outputTokens: 0,
        cacheReadTokens: 0,
        reasoningTokens: 0,
      },
      cost: { amount: 0, unit: 'nanos', currency: 'USD' },
      tags: {},
    });

    expect(parsedEmittedEvent!).toMatchObject(expectedEvent);
    expect(parsedEmittedEvent!.generationId).toHaveLength(36);
  });

  describe('regression tests for narevai/ai-billing#351', () => {
    it('should not double-charge cached input tokens', async () => {
      const destinationSpy = vi.fn();
      const geminiPricing: ModelPricing = {
        promptTokens: 2.0 / 1_000_000, // $2/M
        completionTokens: 12.0 / 1_000_000, // $12/M
        inputCacheReadTokens: 0.2 / 1_000_000, // $0.20/M
        internalReasoningTokens: 12.0 / 1_000_000, // $12/M
      };

      const middleware = createGoogleV4Middleware({
        destinations: [destinationSpy],
        priceResolver: vi.fn().mockResolvedValue(geminiPricing),
      });

      const cachedResult = createResult({
        providerMetadata: {
          google: {
            usageMetadata: {
              promptTokenCount: 44621, // total including cached
              candidatesTokenCount: 5, // text output
              thoughtsTokenCount: 213, // reasoning
              cachedContentTokenCount: 42567,
            },
          },
        },
      });

      const mockModel = new MockLanguageModelV4({
        modelId: 'gemini-3.1-pro-preview',
        provider: 'google',
        doGenerate: async () => cachedResult,
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
      await generateText({ model: wrappedModel, prompt: 'Test' });

      await vi.waitFor(() => expect(destinationSpy).toHaveBeenCalledTimes(1));
      const emittedEvent = destinationSpy.mock.calls[0]![0] as BillingEvent;

      // Verify usage reports correct uncached tokens
      // Uncached = 44621 - 42567 = 2054
      expect(emittedEvent.usage.inputTokens).toBe(2054);
      expect(emittedEvent.usage.cacheReadTokens).toBe(42567);
      expect(emittedEvent.usage.outputTokens).toBe(218); // 5 + 213
      expect(emittedEvent.usage.reasoningTokens).toBe(213);

      // Correct cost calculation (from narevai/ai-billing#351):
      // Uncached input: 2054 × $2/M = $0.004108 = 4,108,000 nanos
      // Cached input: 42567 × $0.20/M = $0.0085134 = 8,513,400 nanos
      // Text output: 5 × $12/M = $0.00006 = 60,000 nanos (completion count - reasoning)
      // Reasoning: 213 × $12/M = $0.002556 = 2,556,000 nanos
      // Total: $0.0152374 = 15,237,400 nanos
      expect(emittedEvent.cost?.amount).toBe(15237400);
    });

    it('should handle calls with no cached tokens correctly', async () => {
      const destinationSpy = vi.fn();
      const geminiPricing: ModelPricing = {
        promptTokens: 2.0 / 1_000_000,
        completionTokens: 12.0 / 1_000_000,
        inputCacheReadTokens: 0.2 / 1_000_000,
        internalReasoningTokens: 12.0 / 1_000_000,
      };

      const middleware = createGoogleV4Middleware({
        destinations: [destinationSpy],
        priceResolver: vi.fn().mockResolvedValue(geminiPricing),
      });

      const noCacheResult = createResult({
        providerMetadata: {
          google: {
            usageMetadata: {
              promptTokenCount: 1000,
              candidatesTokenCount: 50,
              thoughtsTokenCount: 100,
              cachedContentTokenCount: 0,
            },
          },
        },
      });

      const mockModel = new MockLanguageModelV4({
        modelId: 'gemini-3.1-pro-preview',
        provider: 'google',
        doGenerate: async () => noCacheResult,
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
      await generateText({ model: wrappedModel, prompt: 'Test' });

      await vi.waitFor(() => expect(destinationSpy).toHaveBeenCalledTimes(1));
      const emittedEvent = destinationSpy.mock.calls[0]![0] as BillingEvent;

      // No cache means all 1000 input tokens are uncached
      expect(emittedEvent.usage.inputTokens).toBe(1000);
      expect(emittedEvent.usage.cacheReadTokens).toBe(0);

      // Cost:
      // Input: 1000 × $2/M = 2,000,000 nanos
      // Text output: 50 × $12/M = 600,000 nanos
      // Reasoning: 100 × $12/M = 1,200,000 nanos
      // Total: 3,800,000 nanos
      expect(emittedEvent.cost?.amount).toBe(3800000);
    });
  });
});
