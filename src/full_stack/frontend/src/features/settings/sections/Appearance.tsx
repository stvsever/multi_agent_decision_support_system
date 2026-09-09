/**
 * Look and feel.
 *
 * Every control here writes to the local appearance mirror first, so the
 * change is visible on the same frame, and is persisted in the same breath.
 */

import { Field, Segmented, Select } from '@/components/ui/primitives'
import { setLocale } from '@/lib/format'
import { useApp } from '@/lib/store'
import type { Accent, Density, Theme } from '@/lib/types'
import { Grid, Group, SectionHead, SliderControl, SwitchRow, TextControl } from '../controls'
import { useSettingsController } from '../state'

/* The chips paint the six accent identities at once, so each one needs its own
   token rather than the single --accent that is live at the time. */
const ACCENTS: { value: Accent; label: string }[] = [
  { value: 'indigo', label: 'Indigo' },
  { value: 'violet', label: 'Violet' },
  { value: 'teal', label: 'Teal' },
  { value: 'amber', label: 'Amber' },
  { value: 'rose', label: 'Rose' },
  { value: 'slate', label: 'Slate' },
]

export function AppearanceSection() {
  const { config, update } = useSettingsController()
  const appearance = config.appearance
  const setAppearance = useApp((s) => s.setAppearance)

  const setTheme = (theme: Theme) => {
    setAppearance({ theme })
    update('appearance', { theme })
  }
  const setAccent = (accent: Accent) => {
    setAppearance({ accent })
    update('appearance', { accent })
  }
  const setDensity = (density: Density) => {
    setAppearance({ density })
    update('appearance', { density })
  }

  return (
    <>
      <SectionHead title="Appearance" description="How the interface looks and how much it moves." />

      <Group title="Theme">
        <Segmented
          value={appearance.theme}
          options={[
            { value: 'system', label: 'System' },
            { value: 'light', label: 'Light' },
            { value: 'dark', label: 'Dark' },
          ]}
          onChange={setTheme}
        />
        <div className="settings__preview">
          <span className="eyebrow">Preview</span>
          <div className="row gap-3 wrap">
            <span className="t-h3">Deep phenotype</span>
            <span className="badge badge--accent">accent</span>
            <span className="badge badge--positive">complete</span>
            <span className="badge badge--caution">iterating</span>
          </div>
          <span className="t-small secondary">
            Body text at the current scale, with a{' '}
            <span className="semibold" style={{ color: 'var(--accent)' }}>
              highlighted phrase
            </span>{' '}
            and a <span className="mono t-tiny">monospaced id</span>.
          </span>
        </div>
      </Group>

      <Group title="Accent">
        <div className="settings__swatches" role="group" aria-label="Accent colour">
          {ACCENTS.map((option) => (
            <button
              key={option.value}
              type="button"
              className="settings__swatch"
              title={option.label}
              aria-label={option.label}
              aria-pressed={appearance.accent === option.value}
              onClick={() => setAccent(option.value)}
            >
              <span style={{ background: `var(--swatch-${option.value})` }} />
            </button>
          ))}
        </div>
      </Group>

      <Group title="Layout">
        <Grid>
          <Field label="Density" hint="Compact tightens spacing and control heights across every screen.">
            <Segmented
              value={appearance.density}
              options={[
                { value: 'comfortable', label: 'Comfortable' },
                { value: 'compact', label: 'Compact' },
              ]}
              onChange={setDensity}
            />
          </Field>
          <Field label={`Font scale: ${appearance.font_scale.toFixed(2)}x`} hint="Scales the whole type ramp. 0.85 to 1.3.">
            <SliderControl
              value={appearance.font_scale}
              min={0.85}
              max={1.3}
              step={0.05}
              presets={[0.85, 1, 1.15, 1.3]}
              format={(value) => `${value.toFixed(2)}x`}
              onCommit={(next) => {
                setAppearance({ fontScale: next })
                update('appearance', { font_scale: next })
              }}
            />
          </Field>
        </Grid>

        <SwitchRow
          label="Reduced motion"
          hint="Removes transitions and animation. The system preference is honoured either way."
          checked={appearance.reduced_motion}
          onChange={(next) => {
            setAppearance({ reducedMotion: next })
            update('appearance', { reduced_motion: next })
          }}
        />
        <SwitchRow
          label="Show advanced by default"
          hint="Opens every advanced disclosure the first time a screen is drawn."
          checked={appearance.show_advanced_by_default}
          onChange={(next) => update('appearance', { show_advanced_by_default: next })}
        />
      </Group>

      <Group title="Flow canvas">
        <SwitchRow
          label="Animate edges"
          hint="Dispatch edges pulse while a step is in flight."
          checked={appearance.flow_animate_edges}
          onChange={(next) => update('appearance', { flow_animate_edges: next })}
        />
        <Grid columns={3}>
          <Field label="Direction">
            <Segmented
              value={appearance.flow_direction}
              options={[
                { value: 'LR', label: 'Left to right' },
                { value: 'TB', label: 'Top to bottom' },
              ]}
              onChange={(next) => update('appearance', { flow_direction: next })}
            />
          </Field>
          <Field label="Edge style">
            <Select
              aria-label="Edge style"
              value={appearance.flow_edge_style}
              onChange={(event) =>
                update('appearance', {
                  flow_edge_style: event.target.value as typeof appearance.flow_edge_style,
                })
              }
            >
              <option value="bezier">Bezier</option>
              <option value="smoothstep">Smooth step</option>
              <option value="straight">Straight</option>
            </Select>
          </Field>
          <Field label="Token badges" hint="Shows the token count on each node.">
            <Segmented
              value={appearance.flow_show_tokens ? 'on' : 'off'}
              options={[
                { value: 'on', label: 'Show' },
                { value: 'off', label: 'Hide' },
              ]}
              onChange={(next) => update('appearance', { flow_show_tokens: next === 'on' })}
            />
          </Field>
        </Grid>
      </Group>

      <Group title="Numbers">
        <Field
          label="Numeric locale"
          hint="A BCP 47 tag such as en-US, en-GB, de-DE, or nl-BE. It sets thousands separators and date format."
        >
          <TextControl
            ariaLabel="Numeric locale"
            value={appearance.numeric_locale}
            onCommit={(next) => {
              const tag = next.trim() || 'en-US'
              setLocale(tag)
              setAppearance({ numericLocale: tag })
              update('appearance', { numeric_locale: tag })
            }}
          />
        </Field>
      </Group>
    </>
  )
}
