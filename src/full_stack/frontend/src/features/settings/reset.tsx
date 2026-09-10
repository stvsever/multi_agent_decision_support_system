/**
 * Putting every saved preference back to its shipped default.
 *
 * One implementation, rendered in two places: About, which always renders, and
 * the configuration row of the storage listing, which only appears when the
 * service can report storage. The reset itself is POST /api/settings/reset and
 * has nothing to do with the storage endpoint, so it is never gated on it.
 */

import { useQueryClient } from '@tanstack/react-query'
import { RotateCcw } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/primitives'
import { ApiError, api } from '@/lib/api'
import { useApp } from '@/lib/store'

/**
 * What a reset costs, in the reader's terms rather than the file's.
 *
 * The service writes a default configuration document over the stored one.
 * Credentials live in a separate private file and prompt overrides in their own
 * store, so neither is touched; the data roots are part of the configuration,
 * so they are the one thing here a reader would not expect to lose.
 */
export const RESET_LOSS =
  'Every saved preference returns to its shipped default: the backend and the models, engine and budget limits, cost guardrails, appearance, and the list of data root folders. Stored credentials are not touched, prompt overrides are left alone, and nothing already written to disk is deleted.'

/** The reset call itself, with the toast and the cache invalidation it implies. */
export function useSettingsReset(): { reset: () => Promise<boolean>; busy: boolean } {
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)
  const setAppearance = useApp((s) => s.setAppearance)
  const [busy, setBusy] = useState(false)

  const reset = async (): Promise<boolean> => {
    setBusy(true)
    try {
      const result = await api.settings.reset()
      // The appearance mirror is applied to the document root, so it has to be
      // told directly; every other section is read back through the cache.
      setAppearance({
        theme: result.config.appearance.theme,
        accent: result.config.appearance.accent,
        density: result.config.appearance.density,
        fontScale: result.config.appearance.font_scale,
        reducedMotion: result.config.appearance.reduced_motion,
        numericLocale: result.config.appearance.numeric_locale,
      })
      await client.invalidateQueries()
      notify({
        tone: 'positive',
        title: 'Settings reset',
        body: 'Every section is back to its shipped default. Stored credentials and prompt overrides were not touched.',
      })
      return true
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'Settings were not reset',
        body: error instanceof ApiError ? error.message : 'The dashboard service rejected the request.',
      })
      return false
    } finally {
      setBusy(false)
    }
  }

  return { reset, busy }
}

/**
 * The reset control that always renders.
 *
 * Two presses, and the sentence between them is the one above: what is lost,
 * not what is stored.
 */
export function ResetAllSettings() {
  const { reset, busy } = useSettingsReset()
  const [armed, setArmed] = useState(false)

  return (
    <div className="settings__store" data-armed={armed}>
      <div className="row gap-3 wrap">
        <span className="stack grow" style={{ gap: 2, minWidth: 0 }}>
          <span className="t-small semibold">Reset all settings</span>
          <span className="t-tiny muted">
            Puts every section of this dialog back to the values COMPASS ships with.
          </span>
        </span>
        {!armed && (
          <Button size="sm" variant="secondary" icon={<RotateCcw size={13} />} onClick={() => setArmed(true)}>
            Reset to defaults
          </Button>
        )}
      </div>

      {armed && (
        <div className="stack gap-2">
          <span className="t-tiny" style={{ color: 'var(--critical)' }}>
            {RESET_LOSS}
          </span>
          <div className="row gap-2">
            <Button size="sm" onClick={() => setArmed(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              size="sm"
              loading={busy}
              onClick={() => {
                void reset().then((done) => {
                  if (done) setArmed(false)
                })
              }}
            >
              Reset everything
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
