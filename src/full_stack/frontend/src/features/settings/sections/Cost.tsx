/** The two thresholds that stand between a plan and an expensive surprise. */

import { OctagonAlert } from 'lucide-react'
import { Callout } from '@/components/ui/primitives'
import { usd } from '@/lib/format'
import { Grid, Group, NumberSetting, SectionHead } from '../controls'
import { useSettingsController } from '../state'

export function CostSection() {
  const { config } = useSettingsController()
  const cost = config.cost

  return (
    <>
      <SectionHead
        title="Cost guardrails"
        description="Checked against the projected spend before a run is allowed to start."
      />

      <Callout tone={cost.block_above_usd > 0 ? 'caution' : 'critical'} icon={<OctagonAlert size={15} />}>
        {cost.block_above_usd > 0 ? (
          <>
            A run whose projection exceeds <span className="semibold">{usd(cost.block_above_usd)}</span> is refused
            before any provider call is made, so nothing is spent. Between {usd(cost.warn_above_usd)} and{' '}
            {usd(cost.block_above_usd)} the estimate is flagged and you confirm it yourself.
          </>
        ) : (
          <>
            The hard stop is disabled. Any projection will be allowed to run, however large. Set a block threshold to
            have the engine refuse a run before it spends anything.
          </>
        )}
      </Callout>

      <Group>
        <Grid columns={3}>
          <NumberSetting
            label="Warn above"
            section="cost"
            field="warn_above_usd"
            min={0}
            max={10_000}
            step={0.5}
            integer={false}
            suffix="USD"
            hint="The estimate is flagged, and the run still starts."
          />
          <NumberSetting
            label="Block above"
            section="cost"
            field="block_above_usd"
            min={0}
            max={100_000}
            step={1}
            integer={false}
            suffix="USD"
            hint="Zero disables the hard stop."
          />
          <NumberSetting
            label="Currency decimals"
            section="cost"
            field="currency_decimals"
            min={2}
            max={6}
            hint="Precision used when a cost is written to a report."
            info={
              <p>
                Per-run costs are often fractions of a cent, so the interface already widens the precision for small
                amounts. This setting controls the persisted figures in the report artifacts.
              </p>
            }
          />
        </Grid>
      </Group>

      <p className="t-small secondary">
        A model with no published price is counted as unknown rather than free, which is why an estimate can carry a
        spread. Self-hosted inference has no per-token price, so neither threshold applies to it.
      </p>
    </>
  )
}
