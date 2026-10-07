/**
 * The Claude Code CLI behind `claudeCode()`: how one AI SDK call becomes one
 * `claude -p` process, and how that process's result becomes the call's
 * content. Pure translation lives here apart from the process, so each half
 * is testable without the other.
 *
 * The CLI runs its own tools, never the caller's, so the caller's tools
 * travel as a structured-output schema: one `anyOf` branch per tool, each
 * with the tool's description and input schema. The CLI hands that schema
 * to the model as the input of its one structured-output tool, and the
 * answer comes back as the tool calls the harness then runs itself.
 */

import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import {
  APICallError,
  type JSONSchema7,
  type LanguageModelV4CallOptions,
  type LanguageModelV4Content,
  type LanguageModelV4FinishReason,
  type LanguageModelV4FunctionTool,
  type LanguageModelV4Message,
  type LanguageModelV4ToolResultOutput,
  type LanguageModelV4Usage,
  type SharedV4Warning,
} from '@ai-sdk/provider';

/** One content block of the user message the CLI reads on stdin. */
export type CliContentBlock =
  | { readonly type: 'text'; readonly text: string }
  | { readonly type: 'image'; readonly source: { readonly type: 'base64'; readonly media_type: string; readonly data: string } };

/** What a call answers with: tool calls, one JSON object, or plain text. */
export type CliAnswer = 'tools' | 'object' | 'text';

/** One AI SDK call, translated for the CLI. */
export interface CliRequest {
  readonly answer: CliAnswer;
  readonly system: string;
  /** The single user message the CLI reads: the whole conversation so far. */
  readonly content: readonly CliContentBlock[];
  /** The structured-output schema; undefined for a plain text answer. */
  readonly schema: JSONSchema7 | undefined;
  /** Output tokens the call may spend, when the caller capped them. */
  readonly maxOutputTokens: number | undefined;
  /** Thinking budget in tokens; 0 is off, as on the API. */
  readonly thinkingTokens: number;
  readonly warnings: readonly SharedV4Warning[];
}

/** The `result` event the CLI prints last in `stream-json` output. */
export interface CliResult {
  readonly subtype: string;
  readonly is_error?: boolean;
  readonly result?: string;
  readonly structured_output?: unknown;
  readonly stop_reason?: string | null;
  readonly session_id?: string;
  readonly total_cost_usd?: number;
  readonly duration_ms?: number;
  readonly duration_api_ms?: number;
  readonly usage?: {
    readonly input_tokens?: number;
    readonly output_tokens?: number;
    readonly cache_read_input_tokens?: number;
    readonly cache_creation_input_tokens?: number;
    readonly output_tokens_details?: { readonly thinking_tokens?: number };
  };
}

/** What one process run produced: the assistant's text before its answer, and the result. */
export interface CliOutput {
  readonly text: string;
  readonly result: CliResult;
}

/** The AI SDK call settings the CLI has no flag for. */
const UNSUPPORTED_SETTINGS = ['temperature', 'topP', 'topK', 'stopSequences', 'seed', 'presencePenalty', 'frequencyPenalty'] as const;

/** Read once per tools call, so the model knows how its earlier turns are replayed. */
const TRANSCRIPT_RULES = [
  'The conversation so far is replayed in one user message: user turns inside <user>, your own earlier turns inside <assistant> with each tool call as <tool_call>, and what each call returned inside <tool_result>. Continue from the end of it.',
  'Answer with the tool calls to make now, in order, as your structured output.',
].join('\n');

/** Translates one AI SDK call into the CLI's input. */
export function toCliRequest(options: LanguageModelV4CallOptions): CliRequest {
  const warnings: SharedV4Warning[] = [];
  for (const setting of UNSUPPORTED_SETTINGS) {
    if (options[setting] !== undefined) warnings.push({ type: 'unsupported', feature: setting });
  }
  const functionTools = (options.tools ?? []).filter((tool): tool is LanguageModelV4FunctionTool => tool.type === 'function');
  if (functionTools.length < (options.tools?.length ?? 0)) {
    warnings.push({ type: 'unsupported', feature: 'provider-defined tools' });
  }
  const choice = options.toolChoice ?? { type: 'auto' };
  const offered = choice.type === 'tool' ? functionTools.filter((tool) => tool.name === choice.toolName) : functionTools;
  const answer: CliAnswer =
    offered.length > 0 && choice.type !== 'none'
      ? 'tools'
      : options.responseFormat?.type === 'json' && options.responseFormat.schema !== undefined
        ? 'object'
        : 'text';
  const systems = options.prompt.flatMap((message) => (message.role === 'system' ? [message.content] : []));
  const conversation = options.prompt.filter((message) => message.role !== 'system');
  const system = (answer === 'tools' ? [...systems, TRANSCRIPT_RULES] : systems).join('\n\n');
  return {
    answer,
    system,
    content: answer === 'tools' || conversation.length > 1 ? transcript(conversation, warnings) : plainUserTurn(conversation, warnings),
    schema:
      answer === 'tools'
        ? toolCallSchema(offered, choice.type !== 'auto')
        : answer === 'object' && options.responseFormat?.type === 'json'
          ? options.responseFormat.schema
          : undefined,
    maxOutputTokens: options.maxOutputTokens,
    thinkingTokens: thinkingBudget(options.providerOptions),
    warnings,
  };
}

/**
 * The tool calls of one turn as a closed grammar: a list whose items are one
 * of the offered tools, each with its own input schema. A forced choice
 * (`required`, or one named tool) needs at least one call.
 */
function toolCallSchema(tools: readonly LanguageModelV4FunctionTool[], forced: boolean): JSONSchema7 {
  return {
    type: 'object',
    properties: {
      tool_calls: {
        type: 'array',
        description: forced
          ? 'The tool calls to make now, in order. At least one.'
          : 'The tool calls to make now, in order. Empty when the answer is the text before this output.',
        ...(forced ? { minItems: 1 } : {}),
        items: { anyOf: tools.map(toolBranch) },
      },
    },
    required: ['tool_calls'],
    additionalProperties: false,
  };
}

function toolBranch(tool: LanguageModelV4FunctionTool): JSONSchema7 {
  // A nested `$schema` is noise to the model; the tool's own keywords stay.
  const { $schema: _dialect, ...input } = tool.inputSchema;
  return {
    type: 'object',
    ...(tool.description === undefined ? {} : { description: tool.description }),
    properties: { name: { const: tool.name }, input },
    required: ['name', 'input'],
    additionalProperties: false,
  };
}

/** `providerOptions.anthropic.thinking`, read the way the API reads it: off unless enabled with a budget. */
function thinkingBudget(providerOptions: LanguageModelV4CallOptions['providerOptions']): number {
  const thinking = providerOptions?.['anthropic']?.['thinking'];
  if (typeof thinking !== 'object' || thinking === null || Array.isArray(thinking)) return 0;
  const { type, budgetTokens } = thinking as { type?: unknown; budgetTokens?: unknown };
  return type === 'enabled' && typeof budgetTokens === 'number' && budgetTokens > 0 ? Math.floor(budgetTokens) : 0;
}

/** A single user turn, sent as it is: what a judgment call looks like. */
function plainUserTurn(conversation: readonly LanguageModelV4Message[], warnings: SharedV4Warning[]): CliContentBlock[] {
  const blocks: CliContentBlock[] = [];
  for (const message of conversation) {
    if (message.role !== 'user') continue;
    for (const part of message.content) {
      if (part.type === 'text') blocks.push({ type: 'text', text: part.text });
      else blocks.push(fileBlock(part.data, part.mediaType, warnings));
    }
  }
  return mergeText(blocks);
}

/** Every turn of the conversation, tagged by who said it, in order. */
function transcript(conversation: readonly LanguageModelV4Message[], warnings: SharedV4Warning[]): CliContentBlock[] {
  const blocks: CliContentBlock[] = [];
  const text = (value: string) => blocks.push({ type: 'text', text: value });
  for (const message of conversation) {
    switch (message.role) {
      case 'user':
        text('<user>');
        for (const part of message.content) {
          if (part.type === 'text') text(part.text);
          else blocks.push(fileBlock(part.data, part.mediaType, warnings));
        }
        text('</user>');
        break;
      case 'assistant':
        text('<assistant>');
        for (const part of message.content) {
          if (part.type === 'text') text(part.text);
          else if (part.type === 'tool-call') text(`<tool_call id="${part.toolCallId}" name="${part.toolName}">${JSON.stringify(part.input)}</tool_call>`);
          else if (part.type === 'tool-result') blocks.push(...toolResult(part.toolCallId, part.toolName, part.output, warnings));
        }
        text('</assistant>');
        break;
      case 'tool':
        for (const part of message.content) {
          if (part.type === 'tool-result') blocks.push(...toolResult(part.toolCallId, part.toolName, part.output, warnings));
        }
        break;
      case 'system':
        break;
    }
  }
  return mergeText(blocks);
}

function toolResult(id: string, name: string, output: LanguageModelV4ToolResultOutput, warnings: SharedV4Warning[]): CliContentBlock[] {
  const failed = output.type === 'error-text' || output.type === 'error-json' || output.type === 'execution-denied';
  const open: CliContentBlock = { type: 'text', text: `<tool_result id="${id}" name="${name}"${failed ? ' error="true"' : ''}>` };
  const close: CliContentBlock = { type: 'text', text: '</tool_result>' };
  switch (output.type) {
    case 'text':
    case 'error-text':
      return [open, { type: 'text', text: output.value }, close];
    case 'json':
    case 'error-json':
      return [open, { type: 'text', text: JSON.stringify(output.value) }, close];
    case 'execution-denied':
      return [open, { type: 'text', text: `denied${output.reason === undefined ? '' : `: ${output.reason}`}` }, close];
    case 'content':
      return [
        open,
        ...output.value.map((item): CliContentBlock => {
          if (item.type === 'text') return { type: 'text', text: item.text };
          if (item.type === 'file') return fileBlock(item.data, item.mediaType, warnings);
          warnings.push({ type: 'unsupported', feature: 'custom tool result content' });
          return { type: 'text', text: '[custom content omitted]' };
        }),
        close,
      ];
  }
}

/** An image becomes an image block; any other file is named, not sent. */
function fileBlock(
  data: { type: string; data?: Uint8Array | string; text?: string },
  mediaType: string,
  warnings: SharedV4Warning[],
): CliContentBlock {
  if (mediaType.startsWith('image/') && data.type === 'data' && data.data !== undefined) {
    const base64 = typeof data.data === 'string' ? data.data : Buffer.from(data.data).toString('base64');
    return { type: 'image', source: { type: 'base64', media_type: mediaType, data: base64 } };
  }
  if (data.type === 'text' && data.text !== undefined) return { type: 'text', text: data.text };
  warnings.push({ type: 'unsupported', feature: `file part (${mediaType}, ${data.type})` });
  return { type: 'text', text: `[${mediaType} file omitted]` };
}

/** Joins adjacent text blocks, so the message is as few blocks as its images allow. */
function mergeText(blocks: readonly CliContentBlock[]): CliContentBlock[] {
  const merged: CliContentBlock[] = [];
  for (const block of blocks) {
    const last = merged.at(-1);
    if (block.type === 'text' && last?.type === 'text') merged[merged.length - 1] = { type: 'text', text: `${last.text}\n${block.text}` };
    else merged.push(block);
  }
  return merged;
}

/** The call's content, finish reason, and usage, from one finished run. */
export function fromCliOutput(
  request: CliRequest,
  output: CliOutput,
  executable: string,
): { content: LanguageModelV4Content[]; finishReason: LanguageModelV4FinishReason; usage: LanguageModelV4Usage } {
  const { result } = output;
  if (result.is_error === true || result.subtype !== 'success') {
    throw cliFailure(result.result ?? `the CLI ended with ${result.subtype}`, executable);
  }
  const usage = toUsage(result);
  const raw = result.stop_reason ?? undefined;
  const text = output.text.trim();
  const lead: LanguageModelV4Content[] = text === '' ? [] : [{ type: 'text', text }];
  switch (request.answer) {
    case 'text':
      return { content: [{ type: 'text', text: result.result ?? output.text }], finishReason: { unified: 'stop', raw }, usage };
    case 'object':
      if (result.structured_output === undefined) throw cliFailure('the CLI returned no structured output', executable);
      return { content: [{ type: 'text', text: JSON.stringify(result.structured_output) }], finishReason: { unified: 'stop', raw }, usage };
    case 'tools': {
      const calls = readToolCalls(result.structured_output);
      if (calls === undefined) throw cliFailure('the CLI returned no tool calls', executable);
      const content: LanguageModelV4Content[] = [
        ...lead,
        ...calls.map((call) => ({
          type: 'tool-call' as const,
          toolCallId: `call_${randomUUID()}`,
          toolName: call.name,
          input: JSON.stringify(call.input ?? {}),
        })),
      ];
      return { content, finishReason: { unified: calls.length > 0 ? 'tool-calls' : 'stop', raw }, usage };
    }
  }
}

/**
 * The `tool_calls` list, or undefined when the output is not shaped like
 * one. A name outside the offered tools passes through: the SDK answers it
 * as an unknown tool, the same refusal the model reads from any provider.
 */
function readToolCalls(output: unknown): { name: string; input: unknown }[] | undefined {
  if (typeof output !== 'object' || output === null) return undefined;
  const calls = (output as { tool_calls?: unknown }).tool_calls;
  if (!Array.isArray(calls)) return undefined;
  const read: { name: string; input: unknown }[] = [];
  for (const call of calls) {
    if (typeof call !== 'object' || call === null) return undefined;
    const { name, input } = call as { name?: unknown; input?: unknown };
    if (typeof name !== 'string') return undefined;
    read.push({ name, input });
  }
  return read;
}

function toUsage(result: CliResult): LanguageModelV4Usage {
  const usage = result.usage ?? {};
  const noCache = usage.input_tokens ?? 0;
  const cacheRead = usage.cache_read_input_tokens ?? 0;
  const cacheWrite = usage.cache_creation_input_tokens ?? 0;
  const output = usage.output_tokens ?? 0;
  const reasoning = usage.output_tokens_details?.thinking_tokens;
  return {
    inputTokens: { total: noCache + cacheRead + cacheWrite, noCache, cacheRead, cacheWrite },
    outputTokens: { total: output, text: reasoning === undefined ? undefined : output - reasoning, reasoning },
  };
}

/**
 * Variables a parent Claude Code session sets for itself: its session id,
 * entrypoint, signed-in user, effort, and feature switches. A `claude`
 * started from a test run inside such a session would inherit them and
 * answer as part of it, so they stay behind. Kept: where the CLI finds its
 * sign-in (`CLAUDE_CONFIG_DIR`, `CLAUDE_CODE_OAUTH_TOKEN`) and which cloud
 * serves it (`CLAUDE_CODE_USE_BEDROCK` and its siblings).
 */
const KEPT_CLAUDE_VARIABLES = /^CLAUDE_(?:CONFIG_DIR|CODE_OAUTH_TOKEN|CODE_USE_[A-Z]+|CODE_SKIP_[A-Z]+_AUTH)$/;

/**
 * The CLI's environment. The parent's, minus a parent Claude Code session's
 * own variables and `ANTHROPIC_API_KEY`, which the CLI would bill instead of
 * the plan it is signed in with; then this call's limits, then the caller's
 * `env` on top.
 */
export function cliEnvironment(
  parent: NodeJS.ProcessEnv,
  request: Pick<CliRequest, 'thinkingTokens' | 'maxOutputTokens'>,
  extra: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(parent)) {
    if (value === undefined || key === 'ANTHROPIC_API_KEY' || key === 'MAX_THINKING_TOKENS' || key === 'CLAUDECODE') continue;
    if (key.startsWith('CLAUDE_') && !KEPT_CLAUDE_VARIABLES.test(key)) continue;
    env[key] = value;
  }
  env['MAX_THINKING_TOKENS'] = String(request.thinkingTokens);
  if (request.maxOutputTokens !== undefined) env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = String(request.maxOutputTokens);
  return { ...env, ...extra };
}

/** Flags every call runs with: no tools, MCP servers, customizations, or saved session of the CLI's own. */
const ISOLATION_FLAGS = ['--tools', '', '--strict-mcp-config', '--safe-mode', '--no-session-persistence'] as const;

/**
 * Linux caps one argument at 128 KiB, and `--json-schema` takes the schema
 * inline only. A larger schema fails here with its size named rather than
 * as a spawn error.
 */
const MAX_SCHEMA_BYTES = 120_000;

/** The CLI's arguments for one call. */
export function cliArguments(
  request: Pick<CliRequest, 'schema'>,
  settings: { readonly modelId: string; readonly systemPromptFile: string; readonly effort: string | undefined },
): string[] {
  const schema = request.schema === undefined ? undefined : JSON.stringify(request.schema);
  if (schema !== undefined && Buffer.byteLength(schema) > MAX_SCHEMA_BYTES) {
    throw new APICallError({
      message: `the tool schema is ${Buffer.byteLength(schema)} bytes, over the ${MAX_SCHEMA_BYTES} the CLI takes inline; offer fewer or smaller tools`,
      url: 'claude-code:cli',
      requestBodyValues: undefined,
      isRetryable: false,
    });
  }
  return [
    '--print',
    '--input-format',
    'stream-json',
    '--output-format',
    'stream-json',
    '--verbose',
    '--model',
    settings.modelId,
    '--system-prompt-file',
    settings.systemPromptFile,
    ...ISOLATION_FLAGS,
    ...(settings.effort === undefined ? [] : ['--effort', settings.effort]),
    ...(schema === undefined ? [] : ['--json-schema', schema]),
  ];
}

/** The one `stream-json` user message the CLI reads on stdin. */
export function cliInput(request: Pick<CliRequest, 'content'>): string {
  return `${JSON.stringify({ type: 'user', message: { role: 'user', content: request.content } })}\n`;
}

/**
 * Runs the CLI once and reads its `stream-json` events: the assistant's text
 * and the closing `result`. Abort kills the process, so a step deadline
 * leaves no `claude` running behind it.
 */
export function runCli(run: {
  readonly executable: string;
  readonly args: readonly string[];
  readonly input: string;
  readonly env: Readonly<Record<string, string>>;
  readonly cwd: string;
  readonly signal: AbortSignal | undefined;
}): Promise<CliOutput> {
  return new Promise<CliOutput>((resolve, reject) => {
    if (run.signal?.aborted === true) {
      reject(run.signal.reason as Error);
      return;
    }
    const child = spawn(run.executable, run.args, {
      cwd: run.cwd,
      env: run.env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    const texts: string[] = [];
    let result: CliResult | undefined;
    const readLine = (line: string) => {
      const event = parseEvent(line);
      if (event?.['type'] === 'assistant') texts.push(...assistantText(event));
      else if (event?.['type'] === 'result') result = event as unknown as CliResult;
    };
    const onAbort = () => {
      child.kill();
      reject(run.signal?.reason as Error);
    };
    run.signal?.addEventListener('abort', onAbort, { once: true });
    child.stdout.setEncoding('utf8').on('data', (chunk: string) => {
      stdout += chunk;
      let newline = stdout.indexOf('\n');
      while (newline !== -1) {
        readLine(stdout.slice(0, newline));
        stdout = stdout.slice(newline + 1);
        newline = stdout.indexOf('\n');
      }
    });
    child.stderr.setEncoding('utf8').on('data', (chunk: string) => {
      stderr = `${stderr}${chunk}`.slice(-4096);
    });
    child.on('error', (error: NodeJS.ErrnoException) => {
      run.signal?.removeEventListener('abort', onAbort);
      reject(
        error.code === 'ENOENT'
          ? new APICallError({
              message: `the Claude Code CLI was not found as "${run.executable}"; install it and sign in by running \`claude\`, or pass its path as claudeCode(model, { executable })`,
              url: 'claude-code:cli',
              requestBodyValues: undefined,
              isRetryable: false,
              cause: error,
            })
          : error,
      );
    });
    child.on('close', (code) => {
      run.signal?.removeEventListener('abort', onAbort);
      if (stdout.trim() !== '') readLine(stdout);
      if (result !== undefined) resolve({ text: texts.join('\n'), result });
      else reject(cliFailure(stderr.trim() || `the CLI exited with code ${code} and no result`, run.executable));
    });
    child.stdin.on('error', () => {
      // The process exited before reading its input; `close` reports why.
    });
    child.stdin.end(run.input);
  });
}

function parseEvent(line: string): Record<string, unknown> | undefined {
  try {
    const event: unknown = JSON.parse(line);
    return typeof event === 'object' && event !== null ? (event as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}

function assistantText(event: Record<string, unknown>): string[] {
  const content = (event['message'] as { content?: unknown } | undefined)?.content;
  if (!Array.isArray(content)) return [];
  return content.flatMap((block: { type?: unknown; text?: unknown }) => (block.type === 'text' && typeof block.text === 'string' ? [block.text] : []));
}

/**
 * A failed run as the SDK's provider error, so the runner's retries and
 * error mapping treat it like any provider's. Overload and rate limits are
 * worth a retry; a plan's usage limit and a missing sign-in are not.
 */
function cliFailure(message: string, executable: string): APICallError {
  const failure = (statusCode: number | undefined, isRetryable: boolean, hint = '') =>
    new APICallError({
      message: `${message}${hint}`,
      url: 'claude-code:cli',
      requestBodyValues: undefined,
      ...(statusCode === undefined ? {} : { statusCode }),
      isRetryable,
    });
  if (/usage limit|limit reached|out of (?:extra )?usage/i.test(message)) return failure(429, false);
  if (/overloaded|\b529\b/i.test(message)) return failure(529, true);
  if (/rate.?limit|\b429\b/i.test(message)) return failure(429, true);
  if (/\/login|not logged in|log in|authenticat|invalid api key|oauth token|\b401\b|\b403\b/i.test(message)) {
    return failure(401, false, ` (sign in by running \`${executable}\`, or \`${executable} setup-token\` for a token)`);
  }
  if (/\b50[0234]\b|timed? ?out|ECONNRESET|socket hang up|network/i.test(message)) return failure(undefined, true);
  return failure(undefined, false);
}
