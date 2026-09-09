/** Defaults applied when a cohort is run rather than a single participant. */

import { Field } from '@/components/ui/primitives'
import { duration } from '@/lib/format'
import { Group, NumberSetting, SectionHead, SliderControl, SwitchRow } from '../controls'
import { useSettingsController } from '../state'

export function BatchSection() {
  const { config, update } = useSettingsController()
  const batch = config.batch

  return (
    <>
      <SectionHead
        title="Batch"
        description="How a cohort of participants is scheduled."
      />

      <Group title="Scheduling">
        <Field
          label={`Concurrency: ${batch.concurrency} participants at once`}
          hint="Each participant is a separate worker process, and each one runs its own plan steps in parallel."
          info={
            <p>
              Batch concurrency multiplies with the executor workers in the Engine section. Two participants at twelve
              workers can mean twenty-four calls in flight, which is where provider rate limits start to bite.
            </p>
          }
        >
          <SliderControl
            value={batch.concurrency}
            min={1}
            max={16}
            presets={[1, 2, 4, 8]}
            onCommit={(next) => update('batch', { concurrency: next })}
          />
        </Field>

        <SwitchRow
          label="Continue on error"
          hint="A failed participant is recorded and the rest of the cohort carries on. Turn this off to stop at the first failure."
          checked={batch.continue_on_error}
          onChange={(next) => update('batch', { continue_on_error: next })}
        />

        <NumberSetting
          label="Per-run timeout"
          section="batch"
          field="run_timeout_seconds"
          min={60}
          max={86_400}
          step={60}
          suffix="s"
          hint={`One participant is abandoned after ${duration(batch.run_timeout_seconds)}. Between one minute and one day.`}
        />
      </Group>
    </>
  )
}
