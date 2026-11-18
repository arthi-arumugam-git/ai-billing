import { generateText, streamText, wrapLanguageModel } from 'ai';
import { describe, expect, it, vi } from 'vitest';
import { createOpenAIV4Middleware } from './language-model-v4-openai-billing-middleware.js';
import {
  BillingEventSchema,
  MockLanguageModelV4,
  convertArrayToReadableStream,
} from '@ai-billing/testing';
import { LanguageModelV4GenerateResult } from '@ai-sdk/provider';
import type { BillingEvent, ModelPricing } from '@ai-billing/types';
import { z } from 'zod';

describe('OpenAIBillingMiddlewareV4 Integration', () => {
  const StrictBillingEventSchema: z.ZodType<BillingEvent> = BillingEventSchema;
  const mockPricing: ModelPricing = {
    promptTokens: 0.0000002, // $0.20 per 1M
    completionTokens: 0.00000125, // $1.25 per 1M
    inputCacheReadTokens: 0.0000001,
    inputCacheWriteTokens: 0,
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
      inputTokens: { total: 13, noCache: 13, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 54, text: 54, reasoning: 0 },
    },
    response: { id: 'resp_0e07e75c3318ef', timestamp: new Date() },
    providerMetadata: {}, // OpenAI uses standard usage block, no metadata needed for basic cost
    ...overrides,
  });

  describe('wrapGenerate', () => {
    it('should extract usage, resolve pricing, calculate cost, and broadcast event', async () => {
      const destinationSpy = vi.fn();
      const middleware = createOpenAIV4Middleware({
        destinations: [destinationSpy],
        priceResolver: mockPriceResolver,
      });

      const baseResult = createResult();
      const mockModel = new MockLanguageModelV4({
        modelId: 'gpt-5',
        provider: 'openai',
        doGenerate: async () => baseResult,
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });

      await generateText({ model: wrappedModel, prompt: 'Capital of France?' });

      expect(mockPriceResolver).toHaveBeenCalledWith({
        modelId: 'gpt-5',
        providerId: 'openai',
      });

      // Calculate expected cost:
      // Prompt: 13 * 0.0000002 * 1e9 = 2,600 nanos
      // Completion: 54 * 0.00000125 * 1e9 = 67,500 nanos
      // Total: 70,100 nanos
      const expectedEvent = StrictBillingEventSchema.parse({
        generationId: baseResult.response?.id,
        modelId: mockModel.modelId,
        provider: mockModel.provider,
        usage: {
          inputTokens: 13,
          outputTokens: 54,
          cacheReadTokens: 0,
          reasoningTokens: 0,
        },
        cost: {
          amount: 70100,
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
      const middleware = createOpenAIV4Middleware({
        destinations: [destinationSpy],
        priceResolver: mockPriceResolver,
      });

      const baseResult = createResult();

      const mockModel = new MockLanguageModelV4({
        modelId: 'gpt-4o',
        provider: 'openai',
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
          inputTokens: 13,
          outputTokens: 54,
          cacheReadTokens: 0,
          reasoningTokens: 0,
        },
        cost: {
          amount: 70100, // 13 * 0.2 + 54 * 1.25 = 70.1 micro-cents -> 70100 nanos
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

  it('should include webSearchCount in usage and cost when sources are in content', async () => {
    const destinationSpy = vi.fn();
    const webSearchPricing: ModelPricing = {
      ...mockPricing,
      webSearch: 0.03, // $0.03 per search = 30,000,000 nanos
    };
    const middleware = createOpenAIV4Middleware({
      destinations: [destinationSpy],
      priceResolver: vi.fn().mockResolvedValue(webSearchPricing),
    });

    const baseResult = createResult({
      content: [
        { type: 'text', text: 'Answer' },
        {
          type: 'source',
          sourceType: 'url',
          id: 'src-1',
          url: 'https://example.com',
        },
        {
          type: 'source',
          sourceType: 'url',
          id: 'src-2',
          url: 'https://example.org',
        },
      ],
    });

    const mockModel = new MockLanguageModelV4({
      modelId: 'gpt-4o-search-preview',
      provider: 'openai',
      doGenerate: async () => baseResult,
    });

    const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
    await generateText({ model: wrappedModel, prompt: 'Latest news?' });

    await vi.waitFor(() => expect(destinationSpy).toHaveBeenCalledTimes(1));

    const emittedPayload = destinationSpy.mock.calls[0]![0] as BillingEvent;
    expect(emittedPayload.usage.webSearchCount).toBe(2);
    // Prompt: 13 * 200 = 2,600 + Completion: 54 * 1,250 = 67,500 + 2 searches * 30,000,000 = 60,000,000
    // Total: 60,070,100 nanos
    expect(emittedPayload.cost?.amount).toBe(60070100);
  });

  it('should omit the cost object entirely if pricing resolves to undefined', async () => {
    const destinationSpy = vi.fn();
    const missingPriceResolver = vi.fn().mockResolvedValue(undefined);

    const middleware = createOpenAIV4Middleware({
      destinations: [destinationSpy],
      priceResolver: missingPriceResolver,
    });

    const baseResult = createResult();
    const mockModel = new MockLanguageModelV4({
      modelId: 'unknown-future-model',
      doGenerate: async () => baseResult,
    });

    const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });

    await generateText({ model: wrappedModel, prompt: 'Hello' });

    const expectedEvent = StrictBillingEventSchema.parse({
      generationId: baseResult.response?.id,
      modelId: mockModel.modelId,
      provider: 'openai',
      usage: {
        inputTokens: 13,
        outputTokens: 54,
        cacheReadTokens: 0,
        reasoningTokens: 0,
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

  it('should hit all fallback branches for full coverage (UUID generation, empty usage)', async () => {
    const destinationSpy = vi.fn();
    const middleware = createOpenAIV4Middleware({
      destinations: [destinationSpy],
      priceResolver: mockPriceResolver,
    });

    const baseResult = createResult({
      response: { id: undefined }, // Forces UUID generation
      usage: {
        inputTokens: {
          total: undefined,
          noCache: undefined,
          cacheRead: undefined,
          cacheWrite: undefined,
        },
        outputTokens: {
          total: undefined,
          text: undefined,
          reasoning: undefined,
        },
      },
    });

    const mockModel = new MockLanguageModelV4({
      modelId: 'gpt-3.5-turbo',
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
      provider: 'openai', // Expected fallback
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
      const gpt5Pricing: ModelPricing = {
        promptTokens: 1.25 / 1_000_000, // $1.25/M
        completionTokens: 10.0 / 1_000_000, // $10/M
        inputCacheReadTokens: 0.125 / 1_000_000, // $0.125/M
      };

      const middleware = createOpenAIV4Middleware({
        destinations: [destinationSpy],
        priceResolver: vi.fn().mockResolvedValue(gpt5Pricing),
      });

      const cachedResult = createResult({
        usage: {
          inputTokens: { total: 14438, noCache: 102, cacheRead: 14336, cacheWrite: 0 },
          outputTokens: { total: 120, text: 56, reasoning: 64 },
        },
      });

      const mockModel = new MockLanguageModelV4({
        modelId: 'gpt-5',
        provider: 'openai',
        doGenerate: async () => cachedResult,
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
      await generateText({ model: wrappedModel, prompt: 'Test' });

      await vi.waitFor(() => expect(destinationSpy).toHaveBeenCalledTimes(1));
      const emittedEvent = destinationSpy.mock.calls[0]![0] as BillingEvent;

      // Verify usage reports correct uncached tokens
      expect(emittedEvent.usage.inputTokens).toBe(102); // uncached only
      expect(emittedEvent.usage.cacheReadTokens).toBe(14336);

      // Correct cost calculation (from narevai/ai-billing#351):
      // Uncached input: 102 × $1.25/M = $0.0001275 = 127,500 nanos
      // Cached input: 14336 × $0.125/M = $0.001792 = 1,792,000 nanos
      // Text output: 56 × $10/M = $0.00056 = 560,000 nanos
      // Reasoning at completion rate: 64 × $10/M = $0.00064 = 640,000 nanos
      // Total: $0.0031195 = 3,119,500 nanos
      expect(emittedEvent.cost?.amount).toBe(3119500);
    });

    it('should use noCache field when available', async () => {
      const destinationSpy = vi.fn();
      const simplePricing: ModelPricing = {
        promptTokens: 1.0 / 1_000_000, // $1/M
        completionTokens: 2.0 / 1_000_000, // $2/M
        inputCacheReadTokens: 0.1 / 1_000_000, // $0.1/M
      };

      const middleware = createOpenAIV4Middleware({
        destinations: [destinationSpy],
        priceResolver: vi.fn().mockResolvedValue(simplePricing),
      });

      const result = createResult({
        usage: {
          inputTokens: { total: 1000, noCache: 200, cacheRead: 800, cacheWrite: 0 },
          outputTokens: { total: 100, text: 100, reasoning: 0 },
        },
      });

      const mockModel = new MockLanguageModelV4({
        modelId: 'test-model',
        provider: 'openai',
        doGenerate: async () => result,
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
      await generateText({ model: wrappedModel, prompt: 'Test' });

      await vi.waitFor(() => expect(destinationSpy).toHaveBeenCalledTimes(1));
      const emittedEvent = destinationSpy.mock.calls[0]![0] as BillingEvent;

      // Verify usage
      expect(emittedEvent.usage.inputTokens).toBe(200); // should use noCache

      // Correct cost:
      // Uncached: 200 × $1/M = 200,000 nanos
      // Cached: 800 × $0.1/M = 80,000 nanos
      // Output: 100 × $2/M = 200,000 nanos
      // Total: 480,000 nanos
      expect(emittedEvent.cost?.amount).toBe(480000);
    });

    it('should fall back to total-cacheRead when noCache is undefined', async () => {
      const destinationSpy = vi.fn();
      const simplePricing: ModelPricing = {
        promptTokens: 1.0 / 1_000_000,
        completionTokens: 2.0 / 1_000_000,
        inputCacheReadTokens: 0.1 / 1_000_000,
      };

      const middleware = createOpenAIV4Middleware({
        destinations: [destinationSpy],
        priceResolver: vi.fn().mockResolvedValue(simplePricing),
      });

      const result = createResult({
        usage: {
          inputTokens: { total: 1000, noCache: undefined, cacheRead: 800, cacheWrite: 0 },
          outputTokens: { total: 100, text: 100, reasoning: 0 },
        },
      });

      const mockModel = new MockLanguageModelV4({
        modelId: 'test-model',
        provider: 'openai',
        doGenerate: async () => result,
      });

      const wrappedModel = wrapLanguageModel({ model: mockModel, middleware });
      await generateText({ model: wrappedModel, prompt: 'Test' });

      await vi.waitFor(() => expect(destinationSpy).toHaveBeenCalledTimes(1));
      const emittedEvent = destinationSpy.mock.calls[0]![0] as BillingEvent;

      // Should compute uncached as total - cacheRead = 1000 - 800 = 200
      expect(emittedEvent.usage.inputTokens).toBe(200);

      // Same cost as above
      expect(emittedEvent.cost?.amount).toBe(480000);
    });
  });
});
