/**
 * `claudeCode('sonnet')`: the Claude Code CLI on this machine as an AI SDK
 * model, so a local run spends the Claude plan `claude` is signed in with
 * rather than an API key. Signing in is the CLI's own (`claude`, then
 * `/login`); nothing here reads or stores a credential.
 *
 * Each model call runs one `claude -p` process, isolated from the CLI's own
 * setup: no built-in tools, MCP servers, CLAUDE.md, skills, hooks, or saved
 * session, and none of a parent Claude Code session's variables. The
 * harness's tools come back as tool calls the harness runs itself, so
 * budgets, secrets, and recording hold exactly as with any other model.
 * Thinking is off unless `providerOptions.anthropic.thinking` enables it,
 * as on the API. The model id is whatever `claude --model` takes: an alias
 * (`sonnet`, `opus`, `opus[1m]` for the 1M-token context) or a full id.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { LanguageModelV4, LanguageModelV4GenerateResult, LanguageModelV4StreamPart } from '@ai-sdk/provider';
import { acquireSlot } from './process-slots.ts';
import { cliArguments, cliEnvironment, cliInput, fromCliOutput, runCli, toCliRequest } from './providers/claude-code.ts';

export interface ClaudeCodeOptions {
  /** The CLI to run. Default `claude`, found on `PATH`. */
  readonly executable?: string;
  /**
   * Most `claude` processes running at once across every test worker on
   * this machine. Default: no cap beyond `workers`, since each worker runs
   * one model call at a time. Time spent waiting for a turn counts against
   * the step's deadline.
   */
  readonly maxConcurrent?: number;
  /** The CLI's `--effort` for every call. Default: the CLI's own. */
  readonly effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max';
  /**
   * Variables set on the CLI's environment after the parent's is cleaned.
   * Pass `ANTHROPIC_API_KEY` here to bill an API key through the CLI
   * instead of the signed-in plan.
   */
  readonly env?: Readonly<Record<string, string>>;
}

/** Where the `maxConcurrent` queue lives, shared by every run on this machine. */
const SLOTS_DIR = join(tmpdir(), 'e2e-claude-code-slots');

export function claudeCode(modelId: string, options: ClaudeCodeOptions = {}): LanguageModelV4 {
  const executable = options.executable ?? 'claude';
  const { maxConcurrent } = options;
  if (maxConcurrent !== undefined && (!Number.isInteger(maxConcurrent) || maxConcurrent < 1)) {
    throw new TypeError(`claudeCode: maxConcurrent must be a positive integer, got ${maxConcurrent}`);
  }

  const doGenerate: LanguageModelV4['doGenerate'] = async (callOptions) => {
    const request = toCliRequest(callOptions);
    const release = maxConcurrent === undefined ? undefined : await acquireSlot(SLOTS_DIR, maxConcurrent, callOptions.abortSignal);
    // An empty working directory: nothing for the CLI to discover there.
    const cwd = await mkdtemp(join(tmpdir(), 'e2e-claude-code-'));
    try {
      const systemPromptFile = join(cwd, 'system.txt');
      await writeFile(systemPromptFile, request.system);
      const output = await runCli({
        executable,
        args: cliArguments(request, { modelId, systemPromptFile, effort: options.effort }),
        input: cliInput(request),
        env: cliEnvironment(process.env, request, options.env),
        cwd,
        signal: callOptions.abortSignal,
      });
      const { content, finishReason, usage } = fromCliOutput(request, output, executable);
      const { result } = output;
      return {
        content,
        finishReason,
        usage,
        warnings: [...request.warnings],
        providerMetadata: {
          'claude-code': {
            ...(result.session_id === undefined ? {} : { sessionId: result.session_id }),
            // What the call would cost at API prices; a plan is not billed per call.
            ...(result.total_cost_usd === undefined ? {} : { apiEquivalentCostUsd: result.total_cost_usd }),
            ...(result.duration_api_ms === undefined ? {} : { durationApiMs: result.duration_api_ms }),
          },
        },
        response: { modelId },
      } satisfies LanguageModelV4GenerateResult;
    } finally {
      release?.();
      await rm(cwd, { recursive: true, force: true });
    }
  };

  return {
    specificationVersion: 'v4',
    provider: 'claude-code',
    modelId,
    supportedUrls: {},
    doGenerate,
    // The CLI answers once, at the end; a stream is the finished answer replayed.
    async doStream(callOptions) {
      const generated = await doGenerate(callOptions);
      return { stream: replay(generated) };
    },
  };
}

function replay(generated: LanguageModelV4GenerateResult): ReadableStream<LanguageModelV4StreamPart> {
  const parts: LanguageModelV4StreamPart[] = [{ type: 'stream-start', warnings: generated.warnings }];
  generated.content.forEach((part, index) => {
    if (part.type === 'text') {
      const id = String(index);
      parts.push({ type: 'text-start', id }, { type: 'text-delta', id, delta: part.text }, { type: 'text-end', id });
    } else if (part.type === 'tool-call') {
      parts.push(part);
    }
  });
  parts.push({
    type: 'finish',
    finishReason: generated.finishReason,
    usage: generated.usage,
    ...(generated.providerMetadata === undefined ? {} : { providerMetadata: generated.providerMetadata }),
  });
  return new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  });
}
