/** How hard the actor-critic loop works and how much of it runs at once. */

import { Field } from '@/components/ui/primitives'
import { Group, SectionHead, SliderControl, SwitchRow } from '../controls'
import { useSettingsController } from '../state'

export function EngineSection() {
  const { config, update } = useSettingsController()
  const engine = config.engine
  const isLocal = config.connection.backend === 'local'

  return (
    <>
      <SectionHead
        title="Engine"
        description="The iteration budget, how much runs in parallel, and what the run reports about itself."
      />

      <Group title="Iterations">
        <Field
          label={`Max iterations: ${engine.max_iterations}`}
          hint="One iteration is plan, execute, integrate, predict, critique."
          info={
            <>
              <p>
                The critic scores every prediction against a task-specific checklist. When it is not satisfied, the
                orchestrator plans again with the critic's objections in hand and the loop repeats.
              </p>
              <p>
                One iteration is the fastest and cheapest possible run and skips that self-correction entirely. Three
                is the shipped balance. Beyond about five the critic rarely changes its mind, so the extra passes
                mostly cost money.
              </p>
            </>
          }
        >
          <SliderControl
            value={engine.max_iterations}
            min={1}
            max={10}
            onCommit={(next) => update('engine', { max_iterations: next })}
            presets={[1, 2, 3, 5]}
          />
        </Field>
      </Group>

      <Group title="Concurrency">
        <Field
          label={`Executor max workers: ${engine.executor_max_workers}`}
          hint={
            isLocal
              ? 'The Local backend runs plan steps one at a time, so this value is ignored while Local is selected.'
              : 'Bounds how many provider calls are in flight at once across unblocked plan steps.'
          }
          info={
            <p>
              Raising this shortens wall-clock time but concentrates spend and can trip provider rate limits. A local
              backend forces sequential execution regardless of what is set here, because one process cannot serve
              parallel generations without thrashing.
            </p>
          }
        >
          <SliderControl
            value={engine.executor_max_workers}
            min={1}
            max={64}
            onCommit={(next) => update('engine', { executor_max_workers: next })}
            presets={[1, 4, 8, 12, 24]}
          />
        </Field>
      </Group>

      <Group title="Behaviour">
        <SwitchRow
          label="Auto repair"
          hint="Retries a failed plan step in place with the error attached, instead of abandoning the step."
          checked={engine.auto_repair_enabled}
          onChange={(next) => update('engine', { auto_repair_enabled: next })}
        />
        <SwitchRow
          label="Detailed tool logging"
          hint="Records every tool prompt and response in the run log. Useful when debugging a tool, noisy otherwise."
          checked={engine.detailed_tool_logging}
          onChange={(next) => update('engine', { detailed_tool_logging: next })}
        />
        <SwitchRow
          label="Verbose"
          hint="Prints stage-by-stage progress from the worker process into the run log."
          checked={engine.verbose}
          onChange={(next) => update('engine', { verbose: next })}
        />
      </Group>
    </>
  )
}
