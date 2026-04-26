/**
 * Provider-agnostic LLM configuration layer.
 *
 * GBrain supports any OpenAI-compatible API (MiniMax, OpenRouter, etc.)
 * alongside the legacy Anthropic + OpenAI hardcoded paths. This module
 * resolves which provider to use based on env vars and config, and
 * constructs an OpenAI client pointed at the right baseURL.
 *
 * Detection order (first match wins):
 *   1. GBRAIN_LLM_PROVIDER env var (explicit override)
 *   2. MINIMAX_API_KEY env var (auto-detect MiniMax)
 *   3. ANTHROPIC_API_KEY env var (legacy Anthropic path)
 *   4. OPENAI_API_KEY env var (fallback to OpenAI for chat)
 *
 * For embeddings, the provider is resolved similarly but MiniMax's
 * embo-01 model may have different dimensions than OpenAI's
 * text-embedding-3-large (1536). The dimensions are configurable.
 */

import OpenAI from 'openai';

// ── Provider presets ────────────────────────────────────────

export interface LLMProviderConfig {
  /** Provider identifier for logging / config. */
  provider: string;
  /** Base URL for the OpenAI-compatible API. */
  baseURL: string;
  /** API key. */
  apiKey: string;
  /** Chat/completion model name. */
  chatModel: string;
  /** Embedding model name. */
  embeddingModel: string;
  /** Embedding vector dimensions. */
  embeddingDimensions: number;
  /** Cheaper/faster model for query expansion (defaults to chatModel). */
  expansionModel: string;
}

const MINIMAX_PRESET: Omit<LLMProviderConfig, 'apiKey'> = {
  provider: 'minimax',
  baseURL: 'https://api.minimax.io/v1',
  chatModel: 'MiniMax-M2.7',
  embeddingModel: 'embo-01',
  embeddingDimensions: 1536, // MiniMax default — override via GBRAIN_EMBEDDING_DIMENSIONS
  expansionModel: 'MiniMax-M2.7',
};

const OPENAI_PRESET: Omit<LLMProviderConfig, 'apiKey'> = {
  provider: 'openai',
  baseURL: 'https://api.openai.com/v1',
  chatModel: 'gpt-4o',
  embeddingModel: 'text-embedding-3-large',
  embeddingDimensions: 1536,
  expansionModel: 'gpt-4o-mini',
};

// ── Singleton state ─────────────────────────────────────────

let resolvedProvider: LLMProviderConfig | null = null;
let openaiClient: OpenAI | null = null;

/**
 * Resolve the active LLM provider. Returns null when no provider
 * is configured (no API keys at all).
 *
 * Result is cached per process. Call `resetLLMProvider()` in tests.
 */
export function getLLMProvider(): LLMProviderConfig | null {
  if (resolvedProvider) return resolvedProvider;

  const explicitProvider = process.env.GBRAIN_LLM_PROVIDER?.toLowerCase();
  const minimaxKey = process.env.MINIMAX_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;
  const anthropicKey = process.env.ANTHROPIC_API_KEY;

  // Custom base URL overrides everything.
  const customBaseURL = process.env.GBRAIN_LLM_BASE_URL;
  const customChatModel = process.env.GBRAIN_CHAT_MODEL;
  const customEmbedModel = process.env.GBRAIN_EMBEDDING_MODEL;
  const customExpansionModel = process.env.GBRAIN_EXPANSION_MODEL;
  const customDimensions = process.env.GBRAIN_EMBEDDING_DIMENSIONS
    ? parseInt(process.env.GBRAIN_EMBEDDING_DIMENSIONS, 10)
    : undefined;

  // Explicit provider override.
  if (explicitProvider === 'minimax' && minimaxKey) {
    resolvedProvider = {
      ...MINIMAX_PRESET,
      apiKey: minimaxKey,
      ...(customBaseURL ? { baseURL: customBaseURL } : {}),
      ...(customChatModel ? { chatModel: customChatModel } : {}),
      ...(customEmbedModel ? { embeddingModel: customEmbedModel } : {}),
      ...(customExpansionModel ? { expansionModel: customExpansionModel } : {}),
      ...(customDimensions ? { embeddingDimensions: customDimensions } : {}),
    };
    return resolvedProvider;
  }

  if (explicitProvider === 'openai' && openaiKey) {
    resolvedProvider = {
      ...OPENAI_PRESET,
      apiKey: openaiKey,
      ...(customBaseURL ? { baseURL: customBaseURL } : {}),
      ...(customChatModel ? { chatModel: customChatModel } : {}),
      ...(customEmbedModel ? { embeddingModel: customEmbedModel } : {}),
      ...(customExpansionModel ? { expansionModel: customExpansionModel } : {}),
      ...(customDimensions ? { embeddingDimensions: customDimensions } : {}),
    };
    return resolvedProvider;
  }

  // Custom provider (OpenRouter, etc.) — needs base URL + key.
  if (explicitProvider === 'custom' && customBaseURL) {
    const key = process.env.GBRAIN_LLM_API_KEY || openaiKey || minimaxKey || '';
    resolvedProvider = {
      provider: 'custom',
      baseURL: customBaseURL,
      apiKey: key,
      chatModel: customChatModel || 'gpt-4o',
      embeddingModel: customEmbedModel || 'text-embedding-3-large',
      embeddingDimensions: customDimensions || 1536,
      expansionModel: customExpansionModel || customChatModel || 'gpt-4o-mini',
    };
    return resolvedProvider;
  }

  // Auto-detect: MiniMax key present → use MiniMax.
  if (minimaxKey) {
    resolvedProvider = {
      ...MINIMAX_PRESET,
      apiKey: minimaxKey,
      ...(customDimensions ? { embeddingDimensions: customDimensions } : {}),
    };
    return resolvedProvider;
  }

  // Fallback: OpenAI key present → use OpenAI for embeddings (chat still uses Anthropic).
  // This is the "legacy" path. getLLMProvider returns the OpenAI config for embeddings;
  // the subagent handler's Anthropic path is separate.
  if (openaiKey) {
    resolvedProvider = {
      ...OPENAI_PRESET,
      apiKey: openaiKey,
      ...(customDimensions ? { embeddingDimensions: customDimensions } : {}),
    };
    return resolvedProvider;
  }

  return null;
}

/**
 * Get an OpenAI SDK client configured for the active provider.
 * The OpenAI SDK works with any OpenAI-compatible API (MiniMax, OpenRouter).
 * Returns null if no provider is configured.
 */
export function getProviderClient(): OpenAI | null {
  const provider = getLLMProvider();
  if (!provider) return null;
  if (openaiClient) return openaiClient;
  openaiClient = new OpenAI({
    baseURL: provider.baseURL,
    apiKey: provider.apiKey,
  });
  return openaiClient;
}

/**
 * Check if the active provider is Anthropic (legacy path).
 * When true, callers should use the Anthropic SDK directly.
 * When false, callers should use the OpenAI-compatible adapter.
 */
export function isAnthropicProvider(): boolean {
  const provider = getLLMProvider();
  // If no provider resolved but Anthropic key exists, it's the legacy path.
  if (!provider && process.env.ANTHROPIC_API_KEY) return true;
  // If a provider was resolved, it's NOT Anthropic (it's OpenAI-compatible).
  return false;
}

/**
 * Check if any embedding provider is available.
 * This replaces the old `!process.env.OPENAI_API_KEY` checks throughout the codebase.
 */
export function hasEmbeddingProvider(): boolean {
  const provider = getLLMProvider();
  return provider !== null;
}

/**
 * Reset singleton state (for tests).
 */
export function resetLLMProvider(): void {
  resolvedProvider = null;
  openaiClient = null;
}

export { MINIMAX_PRESET, OPENAI_PRESET };
