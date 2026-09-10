/**
 * How hard the loop works, how much of it runs at once, and the token ceilings
 * it enforces. Iterations, concurrency, and budgets were three sections that
 * were always read together, so they are one.
 */

import { Disclosure, Field } from '@/components/ui/primitives'
import { duration, tokens } from '@/lib/format'
import { Grid, Group, NumberSetting, SectionHead, SliderControl, SwitchRow } from '../controls'
import { useSettingsController } from '../state'

const DERIVED = 'Zero derives this from the context window of the model in use.'

export function EngineSection() {
  const { config, update } = useSettingsController()
  const engine = config.engine
  const batch = config.batch
  const budget = config.token_budget
  const isLocal = config.connection.backend === 'local'

  return (
    <>
      <SectionHead
        title="Engine"
        description="The defaults every run starts from: how many passes it takes, how much runs at once, and what it may spend in tokens."
      />

      <Group title="Actor-critic loop">
        <Field
          label={`Max iterations: ${engine.max_iterations}`}
          hint="One iteration is plan, execute, integrate, predict, critique."
          info={
            <p>
              The critic scores every prediction against a task checklist. When it is not satisfied, the orchestrator
              plans again with the objections in hand. One iteration skips that self-correction; three is the shipped
              balance; beyond about five the critic rarely changes its mind.
            </p>
          }
        >
          <SliderControl
            value={engine.max_iterations}
            min={1}
            max={10}
            onCommit={(next) => update('engine', { max_iterations: next })}
          />
        </Field>
        <SwitchRow
          label="Auto repair"
          hint="Retries a failed plan step in place with the error attached, instead of abandoning the step."
          checked={engine.auto_repair_enabled}
          onChange={(next) => update('engine', { auto_repair_enabled: next })}
        />
      </Group>

      <Group title="Concurrency">
        <Field
          label={`Executor workers: ${engine.executor_max_workers}`}
          hint={
            isLocal
              ? 'Ignored while inference is self-hosted: one process cannot serve parallel generations without thrashing.'
              : 'Model calls in flight at once across unblocked plan steps, within one participant.'
          }
        >
          <SliderControl
            value={engine.executor_max_workers}
            min={1}
            max={64}
            onCommit={(next) => update('engine', { executor_max_workers: next })}
          />
        </Field>
        <Field
          label={`Participants at once: ${batch.concurrency}`}
          hint="Applies to a cohort run. Each participant is a separate worker process."
          info={
            <p>
              This multiplies with the executor workers above. Two participants at twelve workers can mean twenty-four
              calls in flight, which is where provider rate limits start to bite.
            </p>
          }
        >
          <SliderControl
            value={batch.concurrency}
            min={1}
            max={16}
            onCommit={(next) => update('batch', { concurrency: next })}
          />
        </Field>
        <SwitchRow
          label="Continue on error"
          hint="A failed participant is recorded and the cohort carries on. Off stops at the first failure."
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
          width={160}
          suffix="s"
          hint={`One participant is abandoned after ${duration(batch.run_timeout_seconds)}.`}
        />
      </Group>

      <Group title="Run log">
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

      <Disclosure
        title="Token budgets"
        subtitle="Ceilings the engine enforces while a run is in flight"
        defaultOpen={config.appearance.show_advanced_by_default}
      >
        <div className="stack gap-5">
          <Group title="Whole run">
            <NumberSetting
              label="Total token budget"
              section="token_budget"
              field="total_budget"
              min={0}
              max={100_000_000}
              step={50_000}
              width={180}
              hint={
                budget.total_budget > 0
                  ? `A run stops once it has consumed ${tokens(budget.total_budget)} tokens across every agent and tool.`
                  : 'Zero removes the run-level ceiling; only the cost guardrails will stop a run.'
              }
            />
          </Group>

          <Group title="Per call">
            <Grid>
              <NumberSetting
                label="Max agent input tokens"
                section="token_budget"
                field="max_agent_input_tokens"
                min={0}
                max={4_000_000}
                step={1000}
                hint={DERIVED}
              />
              <NumberSetting
                label="Max agent output tokens"
                section="token_budget"
                field="max_agent_output_tokens"
                min={0}
                max={1_000_000}
                step={500}
                hint={DERIVED}
              />
              <NumberSetting
                label="Max tool input tokens"
                section="token_budget"
                field="max_tool_input_tokens"
                min={0}
                max={4_000_000}
                step={1000}
                hint={DERIVED}
              />
              <NumberSetting
                label="Max tool output tokens"
                section="token_budget"
                field="max_tool_output_tokens"
                min={0}
                max={1_000_000}
                step={500}
                hint={DERIVED}
              />
            </Grid>
          </Group>

          <Group title="Per component">
            <Grid>
              <NumberSetting
                label="Orchestrator"
                section="token_budget"
                field="orchestrator_budget"
                min={0}
                max={10_000_000}
                step={1000}
                hint="Writing the execution plan."
              />
              <NumberSetting
                label="Executor per step"
                section="token_budget"
                field="executor_budget_per_step"
                min={0}
                max={10_000_000}
                step={1000}
                hint="One tool step, including any repair attempt."
              />
              <NumberSetting
                label="Fusion"
                section="token_budget"
                field="fusion_budget"
                min={0}
                max={10_000_000}
                step={1000}
                hint="Merging step outputs before integration."
              />
              <NumberSetting
                label="Integrator"
                section="token_budget"
                field="integrator_budget"
                min={0}
                max={10_000_000}
                step={1000}
                hint="Building the single evidence representation."
              />
              <NumberSetting
                label="Predictor"
                section="token_budget"
                field="predictor_budget"
                min={0}
                max={10_000_000}
                step={1000}
                hint="Producing the phenotype outputs."
              />
              <NumberSetting
                label="Critic"
                section="token_budget"
                field="critic_budget"
                min={0}
                max={10_000_000}
                step={1000}
                hint="Scoring the prediction."
              />
              <NumberSetting
                label="Communicator"
                section="token_budget"
                field="communicator_budget"
                min={0}
                max={10_000_000}
                step={1000}
                hint="Writing the deep phenotype report."
              />
            </Grid>
            <p className="t-tiny muted">
              Every component ceiling defaults to zero, which lets the engine size it from the total budget and the
              context window. Set one only to hold back a stage that is eating the run.
            </p>
          </Group>
        </div>
      </Disclosure>
    </>
  )
}
