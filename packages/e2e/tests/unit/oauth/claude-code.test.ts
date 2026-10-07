import { chmodSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { APICallError, type JSONSchema7, type LanguageModelV4CallOptions } from '@ai-sdk/provider';
import { generateText, jsonSchema, NoObjectGeneratedError, Output, tool } from 'ai';
import { Ajv } from 'ajv';
import { afterEach, describe, expect, it } from 'vitest';
import { claudeCode } from '../../../src/oauth/claude-code.ts';
import { cliEnvironment, toCliRequest, type CliResult } from '../../../src/oauth/providers/claude-code.ts';

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function tempDir(prefix: string): string {
  const dir = mkdtempSync(path.join(tmpdir(), prefix));
  dirs.push(dir);
  return dir;
}

interface FakeCall {
  readonly argv: string[];
  readonly env: Record<string, string>;
  readonly system: string;
  readonly input: { type: string; message: { role: string; content: { type: string; text?: string; source?: { media_type: string } }[] } };
}

/**
 * A stand-in `claude`: records each run's arguments, environment, system
 * prompt, and stdin, then prints the next scripted `stream-json` output.
 * A scripted entry may sleep first, to be aborted.
 */
function fakeCli(outputs: readonly ({ text?: string; result: Partial<CliResult> } | { sleepMs: number } | { stderr: string; exitCode: number })[]) {
  const dir = tempDir('e2e-fake-claude-');
  const script = path.join(dir, 'claude');
  writeFileSync(path.join(dir, 'outputs.json'), JSON.stringify(outputs));
  writeFileSync(
    script,
    `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const dir = ${JSON.stringify(dir)};
const index = fs.readdirSync(dir).filter((name) => name.startsWith('call-')).length;
const argv = process.argv.slice(2);
const input = fs.readFileSync(0, 'utf8');
const system = fs.readFileSync(argv[argv.indexOf('--system-prompt-file') + 1], 'utf8');
fs.writeFileSync(path.join(dir, 'call-' + index + '.json'), JSON.stringify({ argv, env: process.env, system, input: JSON.parse(input), pid: process.pid }));
const output = JSON.parse(fs.readFileSync(path.join(dir, 'outputs.json'), 'utf8'))[index];
if (output.sleepMs !== undefined) { setTimeout(() => {}, output.sleepMs); return; }
if (output.stderr !== undefined) { process.stderr.write(output.stderr); process.exit(output.exitCode); }
const print = (event) => process.stdout.write(JSON.stringify(event) + '\\n');
print({ type: 'system', subtype: 'init' });
if (output.text !== undefined) print({ type: 'assistant', message: { content: [{ type: 'text', text: output.text }] } });
print({ type: 'result', subtype: 'success', is_error: false, ...output.result });
`,
  );
  chmodSync(script, 0o755);
  const calls = (): (FakeCall & { pid: number })[] =>
    readdirSync(dir)
      .filter((name) => name.startsWith('call-'))
      .toSorted((a, b) => Number(a.slice(5, -5)) - Number(b.slice(5, -5)))
      .map((name) => JSON.parse(readFileSync(path.join(dir, name), 'utf8')) as FakeCall & { pid: number });
  return { executable: script, calls };
}

const flag = (argv: readonly string[], name: string) => argv[argv.indexOf(name) + 1];
const inputText = (call: FakeCall) => call.input.message.content.map((block) => block.text ?? `[${block.type}]`).join('\n');

const USAGE = { input_tokens: 12, cache_read_input_tokens: 300, cache_creation_input_tokens: 40, output_tokens: 7 };

describe('claudeCode', () => {
  it('runs a tool loop: the tools go out as one schema, come back as tool calls, and the results replay on the next turn', async () => {
    const cli = fakeCli([
      { text: 'Tapping Save.', result: { structured_output: { tool_calls: [{ name: 'tap', input: { id: 'n1' } }] }, usage: USAGE, stop_reason: 'tool_use' } },
      { result: { structured_output: { tool_calls: [{ name: 'complete_step', input: { status: 'passed' } }] }, usage: USAGE } },
    ]);
    const tapped: string[] = [];
    const result = await generateText({
      model: claudeCode('sonnet', { executable: cli.executable, effort: 'low' }),
      system: 'You operate a UI.',
      prompt: 'Screen: #n1 button "Save". Press Save.',
      toolChoice: 'required',
      tools: {
        tap: tool({
          description: 'Taps one node.',
          inputSchema: jsonSchema<{ id: string }>({ type: 'object', properties: { id: { type: 'string' } }, required: ['id'] }),
          execute: async ({ id }) => {
            tapped.push(id);
            return `tapped ${id}`;
          },
        }),
        complete_step: tool({ inputSchema: jsonSchema({ type: 'object', properties: { status: { type: 'string' } } }) }),
      },
      stopWhen: ({ steps }) => steps.at(-1)?.toolCalls.some((call) => call.toolName === 'complete_step') === true,
    });

    expect(tapped).toEqual(['n1']);
    expect(result.steps.map((step) => step.toolCalls.map((call) => call.toolName))).toEqual([['tap'], ['complete_step']]);
    expect(result.steps[0]?.text).toBe('Tapping Save.');
    expect(result.steps[0]?.usage).toMatchObject({ inputTokens: 352, outputTokens: 7, inputTokenDetails: { cacheReadTokens: 300, cacheWriteTokens: 40, noCacheTokens: 12 } });

    const [first, second] = cli.calls();
    expect(first?.argv).toEqual(expect.arrayContaining(['--print', '--tools', '', '--strict-mcp-config', '--safe-mode', '--setting-sources=', '--no-session-persistence']));
    expect(flag(first!.argv, '--model')).toBe('sonnet');
    expect(flag(first!.argv, '--effort')).toBe('low');
    expect(first?.system).toMatch(/^You operate a UI\.\n\nThe conversation so far is replayed/);
    const schema = JSON.parse(flag(first!.argv, '--json-schema')!) as { properties: { tool_calls: { minItems?: number; items: { anyOf: { description?: string; properties: { name: { const: string }; input: unknown } }[] } } } };
    expect(schema.properties.tool_calls.minItems).toBe(1);
    expect(schema.properties.tool_calls.items.anyOf.map((branch) => [branch.properties.name.const, branch.description])).toEqual([
      ['tap', 'Taps one node.'],
      ['complete_step', undefined],
    ]);
    expect(inputText(first!)).toBe('<user>\nScreen: #n1 button "Save". Press Save.\n</user>');
    // The second turn starts with the first turn's text, so the prefix the CLI caches is shared.
    expect(inputText(second!)).toMatch(/^<user>\nScreen: #n1 button "Save"\. Press Save\.\n<\/user>\n<assistant>\nTapping Save\.\n<tool_call id="call_[^"]+" name="tap">\{"id":"n1"\}<\/tool_call>\n<\/assistant>\n<tool_result id="call_[^"]+" name="tap">\ntapped n1\n<\/tool_result>$/);
  });

  it('answers a structured judgment with the caller\'s schema as is, and the prompt unwrapped', async () => {
    const cli = fakeCli([{ result: { structured_output: { verdict: 'pass' }, usage: USAGE, total_cost_usd: 0.002, session_id: 's-1' } }]);
    const schema: JSONSchema7 = { type: 'object', properties: { verdict: { enum: ['pass', 'fail'] } }, required: ['verdict'], additionalProperties: false };
    const result = await generateText({
      model: claudeCode('opus[1m]', { executable: cli.executable }),
      system: 'Judge the claim.',
      prompt: 'Claim: the cart has three items.',
      output: Output.object({ schema: jsonSchema<{ verdict: string }>(schema) }),
    });
    expect(result.output).toEqual({ verdict: 'pass' });
    expect(result.providerMetadata).toEqual({ 'claude-code': { sessionId: 's-1', apiEquivalentCostUsd: 0.002 } });
    const [call] = cli.calls();
    expect(flag(call!.argv, '--model')).toBe('opus[1m]');
    expect(call?.argv).not.toContain('--effort');
    expect(JSON.parse(flag(call!.argv, '--json-schema')!)).toEqual(schema);
    expect(call?.system).toBe('Judge the claim.');
    expect(inputText(call!)).toBe('Claim: the cart has three items.');
  });

  it('starts the CLI clean: no parent session variables, no API key, thinking off unless asked', async () => {
    const parent = {
      PATH: '/bin',
      HOME: '/home/me',
      CLAUDECODE: '1',
      CLAUDE_CODE_SESSION_ID: 'parent-session',
      CLAUDE_CODE_USER_EMAIL: 'me@example.com',
      CLAUDE_EFFORT: 'max',
      MAX_THINKING_TOKENS: '31999',
      ANTHROPIC_API_KEY: 'sk-ant-api',
      ANTHROPIC_BASE_URL: 'https://proxy.example',
      CLAUDE_CONFIG_DIR: '/home/me/.claude-work',
      CLAUDE_CODE_OAUTH_TOKEN: 'sk-ant-oat',
      CLAUDE_CODE_USE_BEDROCK: '1',
    };
    expect(cliEnvironment(parent, { thinking: 0, maxOutputTokens: undefined }, undefined)).toEqual({
      PATH: '/bin',
      HOME: '/home/me',
      ANTHROPIC_BASE_URL: 'https://proxy.example',
      CLAUDE_CONFIG_DIR: '/home/me/.claude-work',
      CLAUDE_CODE_OAUTH_TOKEN: 'sk-ant-oat',
      CLAUDE_CODE_USE_BEDROCK: '1',
      MAX_THINKING_TOKENS: '0',
    });
    // As on the API, the thinking budget comes on top of the answer's.
    expect(cliEnvironment({}, { thinking: 4000, maxOutputTokens: 900 }, { ANTHROPIC_API_KEY: 'sk-mine' })).toEqual({
      MAX_THINKING_TOKENS: '4000',
      CLAUDE_CODE_MAX_OUTPUT_TOKENS: '4900',
      ANTHROPIC_API_KEY: 'sk-mine',
    });
    // Adaptive thinking is the CLI's and the model's to size.
    expect(cliEnvironment({ MAX_THINKING_TOKENS: '9' }, { thinking: 'adaptive', maxOutputTokens: 900 }, undefined)).toEqual({ CLAUDE_CODE_MAX_OUTPUT_TOKENS: '900' });

    const options = (providerOptions: LanguageModelV4CallOptions['providerOptions']): LanguageModelV4CallOptions => ({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      ...(providerOptions === undefined ? {} : { providerOptions }),
    });
    expect(toCliRequest(options(undefined)).thinking).toBe(0);
    expect(toCliRequest(options({ anthropic: { thinking: { type: 'enabled', budgetTokens: 2048 } } })).thinking).toBe(2048);
    expect(toCliRequest(options({ anthropic: { thinking: { type: 'enabled' } } })).thinking).toBe(1024);
    expect(toCliRequest(options({ anthropic: { thinking: { type: 'adaptive' } } })).thinking).toBe('adaptive');
    expect(toCliRequest(options({ anthropic: { thinking: { type: 'disabled' } } })).thinking).toBe(0);

    const cli = fakeCli([{ result: { result: 'hello', usage: USAGE } }]);
    const previous = { CLAUDECODE: process.env['CLAUDECODE'], ANTHROPIC_API_KEY: process.env['ANTHROPIC_API_KEY'] };
    process.env['CLAUDECODE'] = '1';
    process.env['ANTHROPIC_API_KEY'] = 'sk-ant-api';
    try {
      const result = await generateText({ model: claudeCode('haiku', { executable: cli.executable }), prompt: 'Say hello.', maxOutputTokens: 500 });
      expect(result.text).toBe('hello');
    } finally {
      for (const [key, value] of Object.entries(previous)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
    const [call] = cli.calls();
    expect(call?.env).not.toHaveProperty('CLAUDECODE');
    expect(call?.env).not.toHaveProperty('ANTHROPIC_API_KEY');
    expect(call?.env).toMatchObject({ MAX_THINKING_TOKENS: '0', CLAUDE_CODE_MAX_OUTPUT_TOKENS: '500' });
    expect(call?.argv).not.toContain('--json-schema');
  });

  it('sends screenshots as image blocks, from the user turn and from a tool result', () => {
    const png = new Uint8Array([137, 80, 78, 71]);
    const request = toCliRequest({
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'Look.' }, { type: 'file', mediaType: 'image/png', data: { type: 'data', data: png } }] },
        { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'screenshot', input: {} }] },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'c1',
              toolName: 'screenshot',
              output: { type: 'content', value: [{ type: 'text', text: 'viewport' }, { type: 'file', mediaType: 'image/jpeg', data: { type: 'data', data: 'AAAA' } }] },
            },
          ],
        },
        { role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c2', toolName: 'tap', output: { type: 'error-text', value: 'node n9 is not on screen' } }] },
      ],
      tools: [{ type: 'function', name: 'screenshot', inputSchema: { type: 'object' } }],
      temperature: 0,
    });
    expect(request.content.map((block) => (block.type === 'image' ? `image ${block.source.media_type} ${block.source.data}` : block.text))).toEqual([
      '<user>\nLook.',
      'image image/png iVBORw==',
      '</user>\n<assistant>\n<tool_call id="c1" name="screenshot">{}</tool_call>\n</assistant>\n<tool_result id="c1" name="screenshot">\nviewport',
      'image image/jpeg AAAA',
      '</tool_result>\n<tool_result id="c2" name="tap" error="true">\nnode n9 is not on screen\n</tool_result>',
    ]);
    expect(request.warnings).toEqual([{ type: 'unsupported', feature: 'temperature' }]);
  });

  it('offers only the named tool when the choice is forced to one, and none under toolChoice none', () => {
    const tools = [
      { type: 'function' as const, name: 'tap', inputSchema: { type: 'object' as const } },
      { type: 'function' as const, name: 'complete_step', inputSchema: { type: 'object' as const } },
    ];
    const prompt: LanguageModelV4CallOptions['prompt'] = [{ role: 'user', content: [{ type: 'text', text: 'go' }] }];
    const forced = toCliRequest({ prompt, tools, toolChoice: { type: 'tool', toolName: 'complete_step' } });
    expect(JSON.stringify(forced.schema)).toContain('"const":"complete_step"');
    expect(JSON.stringify(forced.schema)).not.toContain('"const":"tap"');
    const auto = toCliRequest({ prompt, tools, toolChoice: { type: 'auto' } });
    expect(auto.schema).toMatchObject({ properties: { tool_calls: { type: 'array' } } });
    expect((auto.schema as { properties: { tool_calls: { minItems?: number } } }).properties.tool_calls.minItems).toBeUndefined();
    expect(toCliRequest({ prompt, tools, toolChoice: { type: 'none' } })).toMatchObject({ answer: 'text', schema: undefined });
  });

  it('reports CLI failures as provider errors: a usage limit or a missing sign-in is final, an overload is retried', async () => {
    const failing = async (message: string) => {
      const cli = fakeCli([{ result: { subtype: 'success', is_error: true, result: message } }]);
      return claudeCode('sonnet', { executable: cli.executable })
        .doGenerate({ prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] })
        .then(() => undefined, (error: unknown) => error);
    };
    const limit = await failing('Claude AI usage limit reached|1760000000');
    expect(APICallError.isInstance(limit) && [limit.statusCode, limit.isRetryable]).toEqual([429, false]);
    const overloaded = await failing('API Error: 529 {"type":"overloaded_error"}');
    expect(APICallError.isInstance(overloaded) && [overloaded.statusCode, overloaded.isRetryable]).toEqual([529, true]);
    const signedOut = await failing('Invalid API key · Please run /login');
    expect(APICallError.isInstance(signedOut) && [signedOut.statusCode, signedOut.isRetryable]).toEqual([401, false]);
    expect((signedOut as Error).message).toContain('sign in by running');

    const crashed = fakeCli([{ stderr: 'Error: --json-schema is not valid JSON', exitCode: 1 }]);
    await expect(claudeCode('sonnet', { executable: crashed.executable }).doGenerate({ prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] })).rejects.toThrow(
      '--json-schema is not valid JSON',
    );
    await expect(
      claudeCode('sonnet', { executable: path.join(tempDir('e2e-no-claude-'), 'claude') }).doGenerate({ prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }),
    ).rejects.toThrow(/Claude Code CLI was not found/);
  });

  it('kills the CLI when the call is aborted', async () => {
    const cli = fakeCli([{ sleepMs: 60_000 }]);
    const controller = new AbortController();
    const call = claudeCode('sonnet', { executable: cli.executable }).doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      abortSignal: controller.signal,
    });
    await expect.poll(() => cli.calls().length).toBe(1);
    controller.abort(new Error('step deadline'));
    await expect(call).rejects.toThrow('step deadline');
    const pid = cli.calls()[0]!.pid;
    await expect.poll(() => alive(pid)).toBe(false);
  });

  it('keeps every tool\'s $refs resolving once the tools share one schema, a recursive input included', () => {
    const tree: JSONSchema7 = {
      type: 'object',
      properties: { label: { type: 'string' }, children: { type: 'array', items: { $ref: '#' } } },
      required: ['label'],
      additionalProperties: false,
    };
    const form: JSONSchema7 = {
      type: 'object',
      properties: { field: { $ref: '#/definitions/field' }, other: { $ref: '#/$defs/field' } },
      required: ['field'],
      definitions: { field: { type: 'string', minLength: 2 } },
      $defs: { field: { type: 'number' } },
    };
    const { schema } = toCliRequest({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'go' }] }],
      tools: [
        { type: 'function', name: 'outline', inputSchema: tree },
        { type: 'function', name: 'fill', inputSchema: form },
      ],
      toolChoice: { type: 'required' },
    });
    const validate = new Ajv({ strict: false }).compile(schema as Record<string, unknown>);
    const calls = (...items: unknown[]) => ({ tool_calls: items });
    expect(validate(calls({ name: 'outline', input: { label: 'a', children: [{ label: 'b', children: [{ label: 'c' }] }] } }))).toBe(true);
    expect(validate(calls({ name: 'outline', input: { label: 'a', children: [{ children: [] }] } }))).toBe(false);
    expect(validate(calls({ name: 'fill', input: { field: 'ok', other: 3 } }))).toBe(true);
    expect(validate(calls({ name: 'fill', input: { field: 'x' } }))).toBe(false);
    expect(validate(calls({ name: 'fill', input: { field: 'ok', other: 'three' } }))).toBe(false);
  });

  it('answers off-grammar when the CLI gave up on the schema, so the caller\'s repair runs instead of a provider failure', async () => {
    const gaveUp = { subtype: 'error_max_structured_output_retries', is_error: true, usage: USAGE };
    const judge = fakeCli([{ text: 'The total reads $41.', result: gaveUp }]);
    const judgment = generateText({
      model: claudeCode('sonnet', { executable: judge.executable }),
      prompt: 'Claim: the total is $42.',
      output: Output.object({ schema: jsonSchema({ type: 'object', properties: { verdict: { type: 'string' } }, required: ['verdict'] }) }),
    });
    await expect(judgment).rejects.toSatisfy((error) => NoObjectGeneratedError.isInstance(error));

    const act = fakeCli([{ text: 'I cannot find it.', result: gaveUp }]);
    const turn = await claudeCode('sonnet', { executable: act.executable }).doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'tap Save' }] }],
      tools: [{ type: 'function', name: 'tap', inputSchema: { type: 'object' } }],
      toolChoice: { type: 'required' },
    });
    expect(turn.content).toEqual([{ type: 'text', text: 'I cannot find it.' }]);
    expect(turn.finishReason).toEqual({ unified: 'other', raw: 'error_max_structured_output_retries' });
  });
});

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
