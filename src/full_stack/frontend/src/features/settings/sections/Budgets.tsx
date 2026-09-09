/** Token ceilings, from the whole run down to a single component. */

import { Disclosure } from '@/components/ui/primitives'
import { tokens } from '@/lib/format'
import { Grid, Group, NumberSetting, SectionHead } from '../controls'
import { useSettingsController } from '../state'

const DERIVED = 'Zero derives this from the context window of the model in use.'

export function BudgetsSection() {
  const { config } = useSettingsController()
  const budget = config.token_budget

  return (
    <>
      <SectionHead
        title="Budgets"
        description="Token ceilings the engine enforces while a run is in flight."
      />

      <Group title="Whole run">
        <NumberSetting
          label="Total token budget"
          section="token_budget"
          field="total_budget"
          min={0}
          max={100_000_000}
          step={50_000}
          hint={
            budget.total_budget > 0
              ? `A run stops once it has consumed ${tokens(budget.total_budget)} tokens across every agent and tool.`
              : 'Zero removes the run-level ceiling entirely; only the cost guardrails will stop a run.'
          }
          info={
            <p>
              This is the accounting limit for one participant, counted across prompt and completion tokens for every
              agent and tool call. It is a stop, not a target: a typical run finishes well below it.
            </p>
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

      <Disclosure
        title="Per component"
        subtitle="Ceilings for one stage of the pipeline"
        defaultOpen={config.appearance.show_advanced_by_default}
      >
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
        <p className="t-tiny muted" style={{ marginTop: 'var(--s-3)' }}>
          Every component ceiling defaults to zero, which lets the engine size it from the total budget and the context
          window. Set one only to hold back a stage that is eating the run.
        </p>
      </Disclosure>
    </>
  )
}
