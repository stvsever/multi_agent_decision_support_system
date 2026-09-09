/** Which provider the engine talks to, and the credential it uses. */

import { CircleAlert, ExternalLink, KeyRound, RefreshCw, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  Badge,
  Button,
  Callout,
  Disclosure,
  Field,
  InfoDot,
  Input,
  Segmented,
} from '@/components/ui/primitives'
import { ApiError, api } from '@/lib/api'
import { usd } from '@/lib/format'
import { queryKeys, useAccount, useCapabilities, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { BackendName, CredentialStatus } from '@/lib/types'
import { Grid, Group, NumberSetting, SectionHead, TextSetting } from '../controls'
import { useSettingsController } from '../state'

const BACKENDS: { value: BackendName; label: string; blurb: string }[] = [
  {
    value: 'openrouter',
    label: 'OpenRouter',
    blurb:
      'One key, every hosted model, with live prices the cost estimate can use. This is the path the engine is tuned for.',
  },
  {
    value: 'openai',
    label: 'OpenAI',
    blurb: 'Talks to OpenAI directly with an OpenAI key. Only OpenAI models are reachable and pricing is not fetched.',
  },
  {
    value: 'local',
    label: 'Local',
    blurb:
      'Runs a model on this machine through vLLM or transformers. Nothing leaves the host, calls run one at a time, and the Local backend section below applies.',
  },
]

export function ConnectionSection() {
  const { config, update } = useSettingsController()
  const capabilities = useCapabilities()
  const links = capabilities.data?.provider_links ?? {}
  const backend = config.connection.backend
  const active = BACKENDS.find((option) => option.value === backend) ?? BACKENDS[0]

  return (
    <>
      <SectionHead
        title="Connection"
        description="Where model calls go, and the credential that authorises them."
      />

      <Group title="Backend">
        <Segmented
          value={backend}
          options={BACKENDS.map((option) => ({ value: option.value, label: option.label }))}
          onChange={(next) => update('connection', { backend: next })}
        />
        <span className="t-small secondary">{active.blurb}</span>
      </Group>

      <CredentialPanel links={links} />

      <Disclosure
        title="Advanced"
        subtitle="Endpoint, request identity, and retry behaviour"
        defaultOpen={config.appearance.show_advanced_by_default}
      >
        <div className="stack gap-4">
          <TextSetting
            label="Base URL"
            section="connection"
            field="openrouter_base_url"
            mono
            placeholder="https://openrouter.ai/api/v1"
            hint="Change this only to reach a proxy or a compatible gateway."
          />
          <Grid>
            <TextSetting
              label="Site URL"
              section="connection"
              field="openrouter_site_url"
              placeholder="https://your-lab.example"
              hint="Sent as the HTTP-Referer header."
            />
            <TextSetting
              label="App name"
              section="connection"
              field="openrouter_app_name"
              placeholder="COMPASS"
              hint="Sent as the X-Title header and shown in provider activity."
            />
          </Grid>
          <Grid columns={3}>
            <NumberSetting
              label="Request timeout"
              section="connection"
              field="request_timeout_seconds"
              min={10}
              max={1800}
              suffix="s"
              hint="10 to 1800 seconds."
            />
            <NumberSetting
              label="Max retries"
              section="connection"
              field="max_retries"
              min={0}
              max={10}
              hint="Retries after a transport or rate-limit failure."
            />
            <NumberSetting
              label="Retry delay"
              section="connection"
              field="retry_delay_seconds"
              min={0}
              max={60}
              step={0.5}
              integer={false}
              suffix="s"
              hint="Waited before the first retry, then backed off."
            />
          </Grid>
        </div>
      </Disclosure>
    </>
  )
}

/* --- Credential ----------------------------------------------------------- */

function CredentialPanel({ links }: { links: Record<string, string> }) {
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)
  const account = useAccount()
  const [key, setKey] = useState('')
  const [saving, setSaving] = useState(false)

  const credential = useSettings().data?.credentials?.openrouter
  const configured = Boolean(credential?.configured)

  const save = async () => {
    const trimmed = key.trim()
    if (!trimmed) return
    setSaving(true)
    try {
      await api.settings.setCredential('openrouter', trimmed)
      setKey('')
      await client.invalidateQueries({ queryKey: queryKeys.settings })
      await client.invalidateQueries({ queryKey: queryKeys.account })
      notify({ tone: 'positive', title: 'API key stored', body: 'The key is written to a private file on this machine.' })
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'The key was not stored',
        body: error instanceof ApiError ? error.message : 'The dashboard service rejected the credential.',
      })
    } finally {
      setSaving(false)
    }
  }

  return (
    <Group title="OpenRouter API key">
      {!configured && (
        <Callout
          tone="info"
          icon={<KeyRound size={15} />}
          title="Add a key to start running"
          action={
            links.openrouter_signup ? (
              <a
                className="btn btn--primary btn--sm"
                href={links.openrouter_signup}
                target="_blank"
                rel="noreferrer"
              >
                <ExternalLink size={13} /> Get a key
              </a>
            ) : undefined
          }
        >
          <span className="row gap-2 wrap">
            <span>Without a credential the engine can inspect data and estimate cost, but no run will start.</span>
            <InfoDot label="How to get an OpenRouter key">
              <span className="semibold">Four steps, about two minutes.</span>
              <ol>
                <li>
                  Create an account at{' '}
                  <a href={links.openrouter_signup} target="_blank" rel="noreferrer">
                    openrouter.ai/sign-up
                  </a>
                  , then add a little credit. Pay-as-you-go is enough; there is no subscription.
                </li>
                <li>
                  Open the{' '}
                  <a href={links.openrouter_keys} target="_blank" rel="noreferrer">
                    API keys page
                  </a>
                  .
                </li>
                <li>Select Create key, name it COMPASS, and leave the credit limit empty unless you want a hard cap.</li>
                <li>Copy the key once it appears and paste it into the field below, then select Save.</li>
              </ol>
              <p>
                The key is written to a private file next to the configuration and is never sent back to this browser.
                You can also export OPENROUTER_API_KEY in the environment instead.
              </p>
            </InfoDot>
          </span>
        </Callout>
      )}

      <Field
        label="Key"
        hint="Stored locally on this machine. It is never displayed again after it is saved."
        info={
          <p>
            Pasting a new key replaces the stored one. Leaving the field empty and selecting Save clears the stored key
            and falls back to the OPENROUTER_API_KEY environment variable, if one is set.
          </p>
        }
      >
        <div className="row gap-2 wrap">
          <Input
            type="password"
            autoComplete="off"
            spellCheck={false}
            placeholder={configured ? 'Paste a new key to replace the stored one' : 'sk-or-v1-...'}
            aria-label="OpenRouter API key"
            value={key}
            onChange={(event) => setKey(event.target.value)}
            style={{ flex: '1 1 260px' }}
          />
          <Button variant="primary" onClick={save} loading={saving} disabled={!key.trim()}>
            Save
          </Button>
          <Button
            icon={<RefreshCw size={13} />}
            loading={account.isFetching}
            onClick={() => void account.refetch()}
          >
            Verify
          </Button>
        </div>
      </Field>

      <StatusRow credential={credential} account={account.data} links={links} />
    </Group>
  )
}

function StatusRow({
  credential,
  account,
  links,
}: {
  credential?: CredentialStatus
  account?: ReturnType<typeof useAccount>['data']
  links: Record<string, string>
}) {
  const configured = Boolean(credential?.configured)
  const valid = Boolean(account?.valid)

  return (
    <div className="settings__switch" style={{ alignItems: 'center' }}>
      <span className="settings__switch-text">
        <span className="row gap-2 wrap">
          {configured ? (
            <Badge tone={valid ? 'positive' : 'caution'}>
              {valid ? 'Key verified' : 'Key stored, not verified'}
            </Badge>
          ) : (
            <Badge tone="critical">No key configured</Badge>
          )}
          {credential?.masked && <span className="mono t-tiny muted">{credential.masked}</span>}
          {configured && (
            <span className="t-tiny muted">
              from {credential?.source === 'environment' ? 'the environment' : 'the stored file'}
            </span>
          )}
        </span>
        {account && account.configured && (
          <span className="row gap-3 wrap t-tiny muted tabular" style={{ marginTop: 2 }}>
            <span>Used {usd(account.usage_usd)}</span>
            {account.limit_usd !== null && account.limit_usd !== undefined && (
              <span>Limit {usd(account.limit_usd)}</span>
            )}
            <span>
              Remaining{' '}
              <span className="semibold" style={{ color: 'var(--text)' }}>
                {usd(account.remaining_usd)}
              </span>
            </span>
            {account.is_free_tier && <Badge tone="info">free tier</Badge>}
          </span>
        )}
        {account && !account.valid && account.reason && (
          <span className="row gap-2 t-tiny" style={{ color: 'var(--critical)', marginTop: 2 }}>
            <CircleAlert size={12} /> {account.reason}
          </span>
        )}
        {account?.catalog_ready === false && account.catalog_error && (
          <span className="t-tiny muted" style={{ marginTop: 2 }}>
            Model catalog unavailable: {account.catalog_error}
          </span>
        )}
      </span>
      {links.openrouter_credits && (
        <a
          className="t-tiny row gap-1"
          href={links.openrouter_credits}
          target="_blank"
          rel="noreferrer"
          style={{ flex: 'none' }}
        >
          <ShieldCheck size={12} /> Credits
        </a>
      )}
    </div>
  )
}
