/**
 * A machine-wide cap on concurrent holders, shared by every process that
 * names the same directory. Test workers are separate processes, so a cap
 * kept in memory would only count one worker's calls; the queue is files
 * instead. Each caller takes a numbered ticket, created exclusively one past
 * the highest it sees, so no two callers share a number and a newcomer
 * always queues behind every ticket already taken. A caller holds once
 * fewer than `limit` live tickets are ahead of it; a ticket only moves
 * forward, so a holder stays within the cap until it releases. The ticket
 * of a process that is gone (a crashed worker) is removed by whoever sees it.
 */

import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

/** How often a waiter looks at the queue again. */
const POLL_MS = 100;

const TICKET = /^\d{12}$/;

/**
 * Waits until this caller holds one of `limit` places under `dir`, and
 * resolves with its release. Rejects with the signal's reason when `signal`
 * aborts first: the wait counts against the caller's deadline like the work
 * it guards.
 */
export async function acquireSlot(dir: string, limit: number, signal: AbortSignal | undefined): Promise<() => void> {
  signal?.throwIfAborted();
  mkdirSync(dir, { recursive: true });
  const ticket = takeTicket(dir);
  const release = () => rmSync(join(dir, ticket), { force: true });
  try {
    while (liveTickets(dir).filter((other) => other < ticket).length >= limit) {
      await delay(POLL_MS, undefined, signal === undefined ? {} : { signal });
    }
    return release;
  } catch (error) {
    release();
    throw signal?.aborted === true ? signal.reason : error;
  }
}

function takeTicket(dir: string): string {
  let next = Math.max(-1, ...readdirSync(dir).filter((name) => TICKET.test(name)).map(Number)) + 1;
  for (;;) {
    const ticket = String(next).padStart(12, '0');
    try {
      writeFileSync(join(dir, ticket), String(process.pid), { flag: 'wx' });
      return ticket;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      next += 1;
    }
  }
}

/** The tickets whose process still runs, removing the rest. */
function liveTickets(dir: string): string[] {
  return readdirSync(dir).filter((ticket) => {
    if (!TICKET.test(ticket)) return false;
    if (running(ownerOf(join(dir, ticket)))) return true;
    rmSync(join(dir, ticket), { force: true });
    return false;
  });
}

/** The pid a ticket holds; undefined while the ticket is still being written, or already gone. */
function ownerOf(path: string): number | undefined {
  try {
    const pid = Number(readFileSync(path, 'utf8'));
    return Number.isInteger(pid) && pid > 0 ? pid : undefined;
  } catch {
    return undefined;
  }
}

function running(pid: number | undefined): boolean {
  if (pid === undefined) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM: the process exists and belongs to another user.
    return (error as NodeJS.ErrnoException).code !== 'ESRCH';
  }
}
