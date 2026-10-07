import { gateway } from 'ai';
import { claudeCode } from 'e2e/oauth/claude-code';

/**
 * The model the testbed's agentic suites run on: `E2E_MODEL` when set, else
 * `fallback`. An id `claude-code/<model>` runs the local `claude` CLI on the
 * plan it is signed in with (`E2E_MODEL=claude-code/sonnet`); any other id
 * goes through the AI Gateway.
 */
export function agentModel(fallback: string) {
  const id = process.env.E2E_MODEL ?? fallback;
  return id.startsWith('claude-code/') ? claudeCode(id.slice('claude-code/'.length)) : gateway(id);
}
