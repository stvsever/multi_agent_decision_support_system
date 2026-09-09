/** Standing guidance appended to the system prompt of one agent, or of all of them. */

import { Callout, Field } from '@/components/ui/primitives'
import { INSTRUCTION_SLOTS, type InstructionSlot } from '@/lib/types'
import { Group, SectionHead, TextareaControl } from '../controls'
import { useSettingsController } from '../state'

const SLOTS: Record<InstructionSlot, { label: string; when: string }> = {
  global: {
    label: 'Global',
    when: 'Injected into every agent and every tool, ahead of the scoped instruction below.',
  },
  orchestrator: {
    label: 'Orchestrator',
    when: 'Injected while the execution plan is being written.',
  },
  executor: {
    label: 'Executor',
    when: 'Injected when a plan step is dispatched, and again on a repair attempt.',
  },
  tools: {
    label: 'Tools',
    when: 'Injected into every analysis tool, on top of the system prompt that tool already carries.',
  },
  integrator: {
    label: 'Integrator',
    when: 'Injected while step outputs are fused into one evidence representation.',
  },
  predictor: {
    label: 'Predictor',
    when: 'Injected while the phenotype outputs and the evidence chain are produced.',
  },
  critic: {
    label: 'Critic',
    when: 'Injected while the prediction is scored against the task checklist.',
  },
  communicator: {
    label: 'Communicator',
    when: 'Injected while the deep phenotype report is written.',
  },
}

export function InstructionsSection() {
  const { config, update } = useSettingsController()
  const instructions = config.instructions

  return (
    <>
      <SectionHead
        title="Instructions"
        description="Standing guidance carried into every run, without touching the system prompts."
      />

      <Callout tone="info">
        Global is combined with the scoped instruction, so an agent sees both. Use this for study conventions, units,
        or vocabulary. Use Prompts when you need to change how a role reasons.
      </Callout>

      <Group>
        {INSTRUCTION_SLOTS.map((slot) => (
          <Field key={slot} label={SLOTS[slot].label} hint={SLOTS[slot].when}>
            <TextareaControl
              ariaLabel={`${SLOTS[slot].label} instruction`}
              value={instructions[slot] ?? ''}
              rows={slot === 'global' ? 5 : 3}
              placeholder={slot === 'global' ? 'Applies everywhere. Keep it short and concrete.' : 'Optional.'}
              onCommit={(next) => update('instructions', { [slot]: next } as Partial<Record<InstructionSlot, string>>)}
            />
          </Field>
        ))}
      </Group>
    </>
  )
}
