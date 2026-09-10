/** Small helpers shared by the batch composer, the monitor, and Configure. */

import type { RunStatus } from '@/lib/types'

export type Tone = 'neutral' | 'accent' | 'positive' | 'caution' | 'critical' | 'info'

export const RUN_TONE: Record<RunStatus, Tone> = {
  queued: 'neutral',
  running: 'accent',
  cancelling: 'caution',
  succeeded: 'positive',
  failed: 'critical',
  cancelled: 'neutral',
}

export const BATCH_TONE: Record<string, Tone> = {
  running: 'accent',
  completed: 'positive',
  failed: 'critical',
  cancelled: 'neutral',
}

/** The service accepts 60 to 86400 seconds; these are the ones worth offering. */
export const TIMEOUT_CHOICES = [900, 1800, 3600, 7200, 14400, 43200, 86400]

export function timeoutLabel(seconds: number): string {
  if (seconds < 3600) {
    const minutes = Math.round(seconds / 60)
    return `${minutes} minute${minutes === 1 ? '' : 's'}`
  }
  const hours = seconds / 3600
  const rounded = Number.isInteger(hours) ? hours : Number(hours.toFixed(1))
  return `${rounded} hour${rounded === 1 ? '' : 's'}`
}

/** One sentence saying exactly what pressing launch will do. */
export function describeQueue(
  count: number,
  concurrency: number,
  continueOnError: boolean,
  timeoutSeconds: number,
): string {
  if (count === 0) return 'Select participants to see what launching would queue.'
  if (count === 1) {
    return `Launching starts one run, stopped if it is still going after ${timeoutLabel(timeoutSeconds)}.`
  }
  const parallel = Math.min(concurrency, count)
  return [
    `Launching queues ${count} runs, ${parallel} at a time, each with its own report.`,
    continueOnError
      ? 'A failure is recorded and the rest continue.'
      : 'The first failure cancels everything still queued.',
    `Any run still going after ${timeoutLabel(timeoutSeconds)} is stopped.`,
  ].join(' ')
}
