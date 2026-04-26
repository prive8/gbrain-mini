/**
 * OpenAI-compatible adapter for the subagent handler.
 *
 * The subagent handler (src/core/minions/handlers/subagent.ts) speaks
 * Anthropic's Messages API format. This adapter implements the same
 * `MessagesClient` interface but internally uses the OpenAI SDK to
 * call any OpenAI-compatible API (MiniMax, OpenRouter, etc.).
 *
 * Translation happens in two directions:
 *   1. Anthropic request params → OpenAI chat completion params
 *   2. OpenAI chat completion response → Anthropic Message response
 *
 * The subagent handler's main loop stays unchanged — it calls
 * `client.create()` and gets back what looks like an Anthropic response.
 */

import OpenAI from 'openai';
import type { ChatCompletionMessageParam, ChatCompletionTool } from 'openai/resources/chat/completions';
import type Anthropic from '@anthropic-ai/sdk';
import type { MessagesClient } from './minions/handlers/subagent.ts';
import { getLLMProvider, getProviderClient } from './llm-provider.ts';

// ── Anthropic → OpenAI param conversion ─────────────────────

/**
 * Convert Anthropic system prompt to OpenAI system message.
 * Anthropic accepts either a string or an array of content blocks.
 */
function convertSystem(system: Anthropic.MessageCreateParamsNonStreaming['system']): string {
  if (!system) return '';
  if (typeof system === 'string') return system;
  if (Array.isArray(system)) {
    return system
      .filter((b: any) => b.type === 'text')
      .map((b: any) => b.text)
      .join('\n');
  }
  return '';
}

/**
 * Convert Anthropic messages to OpenAI messages format.
 *
 * Anthropic uses ContentBlock[] for content (text + tool_use + tool_result).
 * OpenAI uses string content + tool_calls on assistant messages, and
 * separate tool-result messages with role='tool'.
 */
function convertMessages(
  system: string,
  messages: Anthropic.MessageParam[],
): ChatCompletionMessageParam[] {
  const result: ChatCompletionMessageParam[] = [];

  // System message first.
  if (system) {
    result.push({ role: 'system', content: system });
  }

  for (const msg of messages) {
    if (msg.role === 'user') {
      // User messages may contain text OR tool_result blocks.
      const content = msg.content;
      if (typeof content === 'string') {
        result.push({ role: 'user', content });
        continue;
      }
      if (Array.isArray(content)) {
        // Separate text blocks from tool_result blocks.
        const textParts: string[] = [];
        const toolResults: Array<{ tool_use_id: string; content: string; is_error?: boolean }> = [];

        for (const block of content as any[]) {
          if (block.type === 'text') {
            textParts.push(block.text);
          } else if (block.type === 'tool_result') {
            toolResults.push({
              tool_use_id: block.tool_use_id,
              content: typeof block.content === 'string' ? block.content : JSON.stringify(block.content),
              is_error: block.is_error,
            });
          }
        }

        // Emit text as a user message.
        if (textParts.length > 0) {
          result.push({ role: 'user', content: textParts.join('\n') });
        }

        // Emit tool results as separate 'tool' messages.
        for (const tr of toolResults) {
          result.push({
            role: 'tool',
            tool_call_id: tr.tool_use_id,
            content: tr.content,
          } as ChatCompletionMessageParam);
        }
      }
    } else if (msg.role === 'assistant') {
      const content = msg.content;
      if (typeof content === 'string') {
        result.push({ role: 'assistant', content });
        continue;
      }
      if (Array.isArray(content)) {
        const textParts: string[] = [];
        const toolCalls: Array<{
          id: string;
          type: 'function';
          function: { name: string; arguments: string };
        }> = [];

        for (const block of content as any[]) {
          if (block.type === 'text') {
            // Strip <think>...</think> tags from MiniMax reasoning output.
            const text = typeof block.text === 'string'
              ? block.text.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
              : '';
            if (text) textParts.push(text);
          } else if (block.type === 'tool_use') {
            toolCalls.push({
              id: block.id,
              type: 'function',
              function: {
                name: block.name,
                arguments: typeof block.input === 'string' ? block.input : JSON.stringify(block.input),
              },
            });
          }
        }

        const assistantMsg: any = {
          role: 'assistant',
          content: textParts.join('\n') || null,
        };
        if (toolCalls.length > 0) {
          assistantMsg.tool_calls = toolCalls;
        }
        result.push(assistantMsg);
      }
    }
  }

  return result;
}

/**
 * Convert Anthropic tool definitions to OpenAI format.
 */
function convertTools(
  tools?: Array<{ name: string; description: string; input_schema: unknown; cache_control?: unknown }>,
): ChatCompletionTool[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map(t => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema as Record<string, unknown>,
    },
  }));
}

/**
 * Convert Anthropic tool_choice to OpenAI format.
 */
function convertToolChoice(
  choice?: Anthropic.MessageCreateParamsNonStreaming['tool_choice'],
): OpenAI.ChatCompletionToolChoiceOption | undefined {
  if (!choice) return undefined;
  const tc = choice as any;
  if (tc.type === 'auto') return 'auto';
  if (tc.type === 'none') return 'none';
  if (tc.type === 'any') return 'required';
  if (tc.type === 'tool' && tc.name) {
    return { type: 'function', function: { name: tc.name } };
  }
  return 'auto';
}

// ── OpenAI → Anthropic response conversion ──────────────────

/**
 * Convert an OpenAI ChatCompletion response to an Anthropic Message shape.
 * The subagent handler reads `.content` (ContentBlock[]), `.usage`, and `.stop_reason`.
 */
function convertResponse(
  response: OpenAI.ChatCompletion,
  model: string,
): Anthropic.Message {
  const choice = response.choices?.[0];
  if (!choice) {
    return {
      id: response.id || 'msg_' + Date.now(),
      type: 'message',
      role: 'assistant',
      content: [{ type: 'text', text: 'No response from model.' }],
      model,
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    };
  }

  const msg = choice.message;
  const content: any[] = [];

  // Text content — strip <think> tags if present.
  if (msg.content) {
    const cleaned = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    if (cleaned) {
      content.push({ type: 'text', text: cleaned });
    }
  }

  // Tool calls → Anthropic tool_use blocks.
  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      let input: unknown;
      try {
        input = JSON.parse(tc.function.arguments);
      } catch {
        input = { raw: tc.function.arguments };
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      });
    }
  }

  // If no content blocks at all, add an empty text block.
  if (content.length === 0) {
    content.push({ type: 'text', text: '' });
  }

  // Map OpenAI stop reasons to Anthropic stop reasons.
  let stopReason: string = 'end_turn';
  if (choice.finish_reason === 'tool_calls') stopReason = 'tool_use';
  else if (choice.finish_reason === 'length') stopReason = 'max_tokens';
  else if (choice.finish_reason === 'stop') stopReason = 'end_turn';

  return {
    id: response.id || 'msg_' + Date.now(),
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: stopReason as any,
    stop_sequence: null,
    usage: {
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
    },
  };
}

// ── Adapter class ───────────────────────────────────────────

/**
 * OpenAI-compatible MessagesClient adapter.
 *
 * Implements the `MessagesClient` interface from subagent.ts so it can
 * be used as a drop-in replacement for the Anthropic SDK's messages client.
 * Internally uses the OpenAI SDK to call MiniMax / OpenRouter / etc.
 */
export class OpenAICompatibleMessagesClient implements MessagesClient {
  private client: OpenAI;
  private model: string;

  constructor(client: OpenAI, model: string) {
    this.client = client;
    this.model = model;
  }

  async create(
    params: Anthropic.MessageCreateParamsNonStreaming,
    opts?: { signal?: AbortSignal },
  ): Promise<Anthropic.Message> {
    const systemText = convertSystem(params.system);
    const messages = convertMessages(systemText, params.messages);
    const tools = convertTools(params.tools as any);
    const toolChoice = convertToolChoice(params.tool_choice);

    const openaiParams: OpenAI.ChatCompletionCreateParams = {
      model: params.model || this.model,
      messages,
      max_tokens: params.max_tokens,
      ...(tools ? { tools } : {}),
      ...(toolChoice ? { tool_choice: toolChoice } : {}),
    };

    const response = await this.client.chat.completions.create(
      openaiParams,
      opts?.signal ? { signal: opts.signal } : undefined,
    );

    return convertResponse(response, params.model || this.model);
  }
}

/**
 * Create a MessagesClient for the active provider.
 * Returns the OpenAI-compatible adapter when the provider is not Anthropic.
 * Returns null when the provider is Anthropic (caller should use the Anthropic SDK).
 */
export function createProviderMessagesClient(): MessagesClient | null {
  const provider = getLLMProvider();
  if (!provider) return null;
  // If the legacy Anthropic path is active, return null.
  if (provider.provider === 'openai' && process.env.ANTHROPIC_API_KEY) return null;

  const client = getProviderClient();
  if (!client) return null;

  return new OpenAICompatibleMessagesClient(client, provider.chatModel);
}

// ── Exports for testing ─────────────────────────────────────

export const __testing = {
  convertSystem,
  convertMessages,
  convertTools,
  convertToolChoice,
  convertResponse,
};
