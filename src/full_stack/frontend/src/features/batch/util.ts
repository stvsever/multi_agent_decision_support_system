/** Small helpers shared by the batch composer and monitor. */

import type { Capabilities, RunStatus, TaskSpecInput } from '@/lib/types'

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

/**
 * Which requirements of the chosen prediction type the composed task has not
 * met yet. The requirement list comes from `/capabilities`, so a new task
 * shape added on the server is checked here without a frontend change.
 */
export function missingRequirements(
  task: TaskSpecInput,
  types: Capabilities['prediction_types'] | undefined,
): string[] {
  const spec = types?.find((t) => t.value === task.prediction_type)
  if (!spec) return []
  return spec.needs.filter((need) => {
    switch (need) {
      case 'target_label':
        return task.target_label.trim().length === 0
      case 'control_label':
        return task.control_label.trim().length === 0
      case 'class_labels':
        return task.class_labels.length < 2
      case 'regression_outputs':
        return task.regression_outputs.length === 0
      case 'root':
        return !task.root
      default:
        return false
    }
  })
}

export const REQUIREMENT_LABEL: Record<string, string> = {
  target_label: 'a target label',
  control_label: 'a comparator label',
  class_labels: 'at least two class labels',
  regression_outputs: 'at least one regression output',
  root: 'a task tree root',
}

export function describeRequirements(needs: string[]): string {
  const parts = needs.map((need) => REQUIREMENT_LABEL[need] ?? need.replace(/_/g, ' '))
  if (parts.length <= 1) return parts.join('')
  return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
}
