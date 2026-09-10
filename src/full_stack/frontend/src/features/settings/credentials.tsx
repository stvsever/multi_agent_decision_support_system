/**
 * Stored credentials.
 *
 * Both panels write the secret straight to the service, which keeps it in a
 * private file next to the configuration. Nothing is ever read back into the
 * browser, so the field is always empty and the status line is all there is.
 */

import { useQueryClient } from '@tanstack/react-query'
import { CircleAlert, ExternalLink, KeyRound, RefreshCw, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button, Callout, Field, InfoDot, Input } from '@/components/ui/primitives'
import { ApiError, api } from '@/lib/api'
import { usd } from '@/lib/format'
import { queryKeys, useAccount, useSettings } from '@/lib/hooks'
import { useApp } from '@/lib/store'
import type { CredentialProvider, CredentialStatus } from '@/lib/types'
import { Group } from './controls'

function useCredentialSave(provider: CredentialProvider) {
  const client = useQueryClient()
  const notify = useApp((s) => s.notify)
  const [saving, setSaving] = useState(false)

  const save = async (key: string, title: string) => {
    const trimmed = key.trim()
    if (!trimmed) return false
    setSaving(true)
    try {
      await api.settings.setCredential(provider, trimmed)
      await client.invalidateQueries({ queryKey: queryKeys.settings })
      await client.invalidateQueries({ queryKey: queryKeys.account })
      notify({ tone: 'positive', title, body: 'It is written to a private file on this machine.' })
      return true
    } catch (error) {
      notify({
        tone: 'critical',
        title: 'The key was not stored',
        body: error instanceof ApiError ? error.message : 'The dashboard service rejected the credential.',
      })
      return false
    } finally {
      setSaving(false)
    }
  }

  return { save, saving }
}

/* --- OpenRouter ----------------------------------------------------------- */

export function OpenRouterKeyPanel({
  links,
  /** Null drops the heading, for a panel that already sits under one. */
  heading = 'OpenRouter API key',
  /**
   * False while the engine is pointed at something else. The panel keeps every
   * control, because a key still has to be addable, replaceable, verifiable and
   * inspectable from here; only the urgency goes, since nothing is blocked.
   */
  inUse = true,
}: {
  links: Record<string, string>
  heading?: string | null
  inUse?: boolean
}) {
  const account = useAccount()
  const [key, setKey] = useState('')
  const { save, saving } = useCredentialSave('openrouter')

  const credential = useSettings().data?.credentials?.openrouter
  const configured = Boolean(credential?.configured)

  const howTo = (
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
        The key is written to a private file next to the configuration and is never sent back to this browser. You can
        also export OPENROUTER_API_KEY in the environment instead.
      </p>
    </InfoDot>
  )

  return (
    <Group title={heading ?? undefined}>
      {!inUse && (
        <span className="row gap-2 wrap t-small secondary">
          <span>
            The engine does not call OpenRouter while the backend is self-hosted, so nothing here blocks a run. A key
            saved now is ready the moment you switch back.
          </span>
          {!configured && howTo}
        </span>
      )}

      {inUse && !configured && (
        <Callout
          tone="info"
          icon={<KeyRound size={15} />}
          title="Add a key to start running"
          action={
            links.openrouter_signup ? (
              <a className="btn btn--primary btn--sm" href={links.openrouter_signup} target="_blank" rel="noreferrer">
                <ExternalLink size={13} /> Get a key
              </a>
            ) : undefined
          }
        >
          <span className="row gap-2 wrap">
            <span>Without a credential the engine can inspect data and estimate cost, but no run will start.</span>
            {howTo}
          </span>
        </Callout>
      )}

      <Field
        label="Key"
        hint="Stored locally. It is never displayed again after it is saved."
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
          <Button
            variant="primary"
            loading={saving}
            disabled={!key.trim()}
            onClick={async () => {
              if (await save(key, 'API key stored')) setKey('')
            }}
          >
            Save
          </Button>
          <Button icon={<RefreshCw size={13} />} loading={account.isFetching} onClick={() => void account.refetch()}>
            Verify
          </Button>
        </div>
      </Field>

      <AccountRow credential={credential} account={account.data} links={links} />
    </Group>
  )
}

function AccountRow({
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
            <Badge tone={valid ? 'positive' : 'caution'}>{valid ? 'Key verified' : 'Key stored, not verified'}</Badge>
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

/* --- Hugging Face --------------------------------------------------------- */

export function HuggingFaceKeyPanel() {
  const [key, setKey] = useState('')
  const { save, saving } = useCredentialSave('huggingface')
  const credential = useSettings().data?.credentials?.huggingface
  const configured = Boolean(credential?.configured)

  return (
    <Field
      label="Hugging Face token"
      hint="Only gated or private repositories need one. Everything else downloads without it."
      info={
        <p>
          The token is used for the model search and for pulling weights. It is stored next to the configuration and is
          never sent back to this browser. HF_TOKEN in the environment works instead.
        </p>
      }
    >
      <div className="row gap-2 wrap">
        <Input
          type="password"
          autoComplete="off"
          spellCheck={false}
          placeholder={configured ? 'Paste a new token to replace the stored one' : 'hf_...'}
          aria-label="Hugging Face token"
          value={key}
          onChange={(event) => setKey(event.target.value)}
          style={{ flex: '1 1 240px' }}
        />
        <Button
          loading={saving}
          disabled={!key.trim()}
          onClick={async () => {
            if (await save(key, 'Token stored')) setKey('')
          }}
        >
          Save
        </Button>
        {configured ? (
          <Badge tone="positive">Token stored</Badge>
        ) : (
          <Badge outline>No token</Badge>
        )}
        {credential?.masked && <span className="mono t-tiny muted">{credential.masked}</span>}
      </div>
    </Field>
  )
}
