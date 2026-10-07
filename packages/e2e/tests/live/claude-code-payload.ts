/**
 * What `claudeCode()` makes the real `claude` CLI send, read off a local
 * stand-in for the Messages API: one request per call, thinking off, the
 * caller's system prompt and tools, and nothing of a parent Claude Code
 * session (run it from inside one to check that). Needs the CLI installed,
 * no sign-in and no network:
 *   node tests/live/claude-code-payload.ts [model]
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { generateText, jsonSchema, tool } from 'ai';
import { claudeCode } from '../../src/oauth/claude-code.ts';

const modelId = process.argv[2] ?? 'haiku';

interface Captured {
  readonly headers: Record<string, string | string[] | undefined>;
  readonly body: {
    system: { text: string }[];
    messages: { role: string; content: string | { type: string; text?: string }[] }[];
    tools: { name: string; input_schema: unknown }[];
    thinking?: { type: string };
    max_tokens: number;
  };
}

const requests: Captured[] = [];
const server = createServer((request, response) => {
  let body = '';
  request.on('data', (chunk: Buffer) => (body += chunk.toString('utf8')));
  request.on('end', () => {
    if (!request.url?.startsWith('/v1/messages') || request.url.includes('count_tokens')) {
      response.writeHead(404).end();
      return;
    }
    requests.push({ headers: request.headers, body: JSON.parse(body) as Captured['body'] });
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    const answer = JSON.stringify({ tool_calls: [{ name: 'tap', input: { id: 'n1' } }] });
    const events: [string, unknown][] = [
      ['message_start', { type: 'message_start', message: { id: 'msg_1', type: 'message', role: 'assistant', model: modelId, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 10, output_tokens: 1 } } }],
      ['content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'toolu_1', name: 'StructuredOutput', input: {} } }],
      ['content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: answer } }],
      ['content_block_stop', { type: 'content_block_stop', index: 0 }],
      ['message_delta', { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 5 } }],
      ['message_stop', { type: 'message_stop' }],
    ];
    for (const [name, data] of events) response.write(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`);
    response.end();
  });
});
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address() as AddressInfo;

try {
  const model = claudeCode(modelId, {
    env: { ANTHROPIC_BASE_URL: `http://127.0.0.1:${port}`, ANTHROPIC_API_KEY: 'stand-in', NO_PROXY: '127.0.0.1', no_proxy: '127.0.0.1' },
  });
  const result = await generateText({
    model,
    system: 'SYSTEM-MARKER: you operate a UI.',
    prompt: 'USER-MARKER: press Save.',
    toolChoice: 'required',
    tools: { tap: tool({ description: 'Taps one node.', inputSchema: jsonSchema({ type: 'object', properties: { id: { type: 'string' } }, required: ['id'] }) }) },
  });

  assert.deepEqual(result.toolCalls.map((call) => [call.toolName, call.input]), [['tap', { id: 'n1' }]]);
  assert.equal(requests.length, 1, 'one API request per model call');
  const [{ body, headers }] = requests as [Captured];
  assert.notEqual(body.thinking?.type, 'enabled', 'thinking is off unless the call enables it');
  assert.ok(body.system.some((block) => block.text.startsWith('SYSTEM-MARKER')), 'the caller\'s system prompt is sent');
  assert.deepEqual(body.tools.map((t) => t.name), ['StructuredOutput'], 'no built-in tools, only the structured output');
  assert.match(JSON.stringify(body.tools[0]?.input_schema), /"const":"tap"/, 'the caller\'s tools travel in its schema');

  const texts = body.messages.flatMap((message) =>
    typeof message.content === 'string' ? [message.content] : message.content.flatMap((block) => (block.text === undefined ? [] : [block.text])),
  );
  const parentSession = [process.env['CLAUDE_CODE_SESSION_ID'], process.env['CLAUDE_CODE_USER_EMAIL']].filter((value): value is string => value !== undefined && value !== '');
  for (const leaked of parentSession) {
    assert.ok(!JSON.stringify(body).includes(leaked), `a parent session value reached the request: ${leaked}`);
  }
  const added = texts.filter((text) => !text.includes('USER-MARKER'));
  const systemAdded = body.system.filter((block) => !block.text.startsWith('SYSTEM-MARKER'));
  console.log(
    JSON.stringify(
      {
        model: modelId,
        maxTokens: body.max_tokens,
        contextBeta: String(headers['anthropic-beta'] ?? '').includes('context-1m'),
        addedSystemBlocks: systemAdded.map((block) => block.text.slice(0, 80)),
        addedUserBlocks: added.map((text) => `${text.length} chars: ${text.replace(/\s+/g, ' ').slice(0, 80)}`),
        insideClaudeCode: process.env['CLAUDECODE'] === '1',
      },
      null,
      2,
    ),
  );
  console.log('ok');
} finally {
  server.close();
}
