/**
 * Self-hosted inference.
 *
 * Three parts that only make sense together: what this machine actually has,
 * which checkpoint to run, and the command that would start it. The plan is
 * resolved by the service from the same configuration a run would use, so the
 * command shown here is the command a run issues.
 */

import { useQuery } from '@tanstack/react-query'
import { Cpu, Download, Lock, Search, TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState, type RefObject } from 'react'
import {
  Badge,
  Callout,
  CopyButton,
  Field,
  Input,
  Segmented,
  Select,
  Spinner,
} from '@/components/ui/primitives'
import { api } from '@/lib/api'
import { compactNumber, titleCase, tokens } from '@/lib/format'
import { useDebounced, useDeployPlan, useDeployProbe, useHfSearch } from '@/lib/hooks'
import type {
  DeployCommand,
  DeployPlan,
  DeployProbe,
  HfModelDetail,
  HfModelRow,
  LocalBackendConfig,
  LocalRuntime,
  LocalScheduler,
} from '@/lib/types'
import { Grid, Group, NumberSetting, SliderControl, SwitchRow, TextSetting } from './controls'
import { useSettingsController } from './state'

const ENGINES: { value: LocalBackendConfig['engine']; label: string }[] = [
  { value: 'auto', label: 'Auto: vLLM when it imports, transformers otherwise' },
  { value: 'vllm', label: 'vLLM' },
  { value: 'transformers', label: 'Transformers' },
]

const RUNTIMES: { value: LocalRuntime; label: string }[] = [
  { value: 'native', label: 'Native' },
  { value: 'docker', label: 'Docker' },
  { value: 'apptainer', label: 'Apptainer' },
]

const SCHEDULERS: { value: LocalScheduler; label: string }[] = [
  { value: 'none', label: 'Run here' },
  { value: 'slurm', label: 'Submit to SLURM' },
]

const RUNTIME_NOTE: Record<LocalRuntime, string> = {
  native: 'Starts the engine in this Python environment. Nothing is pulled or built.',
  docker: 'Runs the engine inside a container with the GPUs passed through.',
  apptainer: 'Runs the engine from a SIF image, which is what most shared clusters allow.',
}

/* --- What this machine has ------------------------------------------------ */

export function MachineFacts() {
  const probe = useDeployProbe()

  if (probe.isLoading) {
    return (
      <div className="row gap-2">
        <Spinner size={13} /> <span className="t-tiny muted">Checking this machine</span>
      </div>
    )
  }

  if (probe.isError || !probe.data) {
    return <span className="t-tiny muted">This machine could not be inspected, so no hardware is reported.</span>
  }

  const data = probe.data
  const present = [
    data.vllm_installed && 'vLLM',
    data.transformers_installed && 'transformers',
    data.docker_available && 'Docker',
    data.apptainer_available && 'Apptainer',
    data.slurm_available && 'SLURM',
  ].filter(Boolean) as string[]
  const missing = [
    !data.vllm_installed && 'vLLM',
    !data.transformers_installed && 'transformers',
    !data.docker_available && 'Docker',
    !data.apptainer_available && 'Apptainer',
    !data.slurm_available && 'SLURM',
  ].filter(Boolean) as string[]

  return (
    <div className="stack gap-2">
      <span className="row gap-2 wrap t-small">
        <Cpu size={14} style={{ color: 'var(--text-muted)' }} />
        <span className="semibold">
          {data.gpu_count === 0
            ? 'No GPU detected'
            : `${data.gpu_count} GPU${data.gpu_count === 1 ? '' : 's'}${
                data.total_vram_gb ? `, ${Math.round(data.total_vram_gb)} GB total` : ''
              }`}
        </span>
        {!data.cuda_available && <span className="t-tiny muted">CUDA not available</span>}
      </span>
      {data.gpu_names.length > 0 && (
        <span className="t-tiny muted truncate" title={data.gpu_names.join(', ')}>
          {data.gpu_names.join(', ')}
        </span>
      )}
      <span className="row gap-2 wrap">
        {present.map((name) => (
          <Badge key={name} outline mono>
            {name}
          </Badge>
        ))}
      </span>
      {missing.length > 0 && <span className="t-tiny faint">Not found: {missing.join(', ')}</span>}
    </div>
  )
}

/* --- Model search --------------------------------------------------------- */

/**
 * A local checkout is a perfectly ordinary value for a model name, and it is
 * not a repository id: sending it to the lookup builds a nonsense URL and comes
 * back as a failure the reader would misread as a broken model.
 */
export function isFilesystemPath(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed) return false
  return (
    trimmed.startsWith('/') ||
    trimmed.startsWith('~') ||
    trimmed.startsWith('./') ||
    trimmed.startsWith('../') ||
    trimmed.startsWith('\\\\') ||
    /^[A-Za-z]:[\\/]/.test(trimmed)
  )
}

/** Only a two-part repository id can be looked up. */
const isRepoId = (value: string): boolean => {
  const trimmed = value.trim()
  return trimmed.includes('/') && !isFilesystemPath(trimmed)
}

/**
 * One detail lookup costs the service up to four requests to Hugging Face, so
 * a list of rows resolving at once would hammer it. Two at a time, queued, and
 * every result cached by id under react-query, which also collapses the row and
 * the selected-model badges onto one request.
 */
const DETAIL_CONCURRENCY = 2
let inFlight = 0
const queued: (() => void)[] = []

async function gatedDetail(modelId: string): Promise<HfModelDetail> {
  if (inFlight >= DETAIL_CONCURRENCY) await new Promise<void>((resolve) => queued.push(resolve))
  inFlight += 1
  try {
    return await api.hf.detail(modelId)
  } finally {
    inFlight -= 1
    queued.shift()?.()
  }
}

const useHfDetail = (modelId: string, enabled = true) =>
  useQuery({
    queryKey: ['hf-detail', modelId],
    queryFn: () => gatedDetail(modelId),
    enabled: enabled && isRepoId(modelId),
    staleTime: 30 * 60_000,
    gcTime: 60 * 60_000,
    retry: false,
  })

/** True once the element has been scrolled into the list's viewport. */
function useOnScreen<T extends HTMLElement>(ref: RefObject<T | null>): boolean {
  const [seen, setSeen] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element || seen) return
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true)
      return
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) setSeen(true)
      },
      { rootMargin: '80px' },
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref, seen])

  return seen
}

function HfSearchRow({
  row,
  selected,
  onPick,
}: {
  row: HfModelRow
  selected: boolean
  onPick: (id: string) => void
}) {
  const ref = useRef<HTMLButtonElement>(null)
  const onScreen = useOnScreen(ref)
  // Only the rows a reader can actually see spend a lookup.
  const detail = useHfDetail(row.id, onScreen)
  const facts = detail.data && !detail.data.error ? detail.data : undefined

  return (
    <button
      ref={ref}
      type="button"
      className="settings__hfrow"
      aria-current={selected}
      onClick={() => onPick(row.id)}
    >
      <span className="stack grow" style={{ gap: 1, minWidth: 0 }}>
        <span className="truncate mono t-tiny">{row.id}</span>
        <span className="row gap-2 wrap t-micro muted">
          <span className="row gap-1 tabular">
            <Download size={11} /> {compactNumber(row.downloads)}
          </span>
          {facts?.context_length ? <span className="tabular">{tokens(facts.context_length)} ctx</span> : null}
          {facts?.parameter_count ? (
            <span className="tabular">{compactNumber(facts.parameter_count)} params</span>
          ) : null}
          {row.pipeline_tag && <span>{row.pipeline_tag}</span>}
          {row.is_embedding && <span>embedding</span>}
        </span>
      </span>
      {row.gated && (
        <Badge tone="caution">
          <Lock size={10} /> gated
        </Badge>
      )}
    </button>
  )
}

/** The search box and the result list, shared by the two things a run loads. */
function HfSearchList({
  label,
  hint,
  placeholder,
  task,
  selected,
  onPick,
}: {
  label: string
  hint: string
  placeholder: string
  task?: string
  selected: string
  onPick: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const needle = useDebounced(query.trim(), 300)
  const search = useHfSearch({ q: needle, task, limit: 12 }, needle.length > 1)

  const rows = search.data?.models ?? []
  const problem = search.isError
    ? 'The Hugging Face index could not be reached, so search is unavailable. A repository id can still be typed below.'
    : search.data?.error

  return (
    <>
      <Field label={label} hint={hint}>
        <div className="row gap-2">
          <Search size={14} style={{ flex: 'none', color: 'var(--text-faint)' }} />
          <Input
            value={query}
            placeholder={placeholder}
            aria-label={label}
            onChange={(event) => setQuery(event.target.value)}
          />
          {search.isFetching && <Spinner size={14} />}
        </div>
      </Field>

      {problem && <span className="t-tiny muted">{problem}</span>}

      {needle.length > 1 && !problem && rows.length === 0 && !search.isFetching && (
        <span className="t-tiny muted">Nothing matched that search.</span>
      )}

      {rows.length > 0 && (
        <div className="settings__hflist">
          {rows.map((row) => (
            <HfSearchRow
              key={row.id}
              row={row}
              selected={row.id === selected}
              onPick={(id) => {
                onPick(id)
                setQuery('')
              }}
            />
          ))}
        </div>
      )}
    </>
  )
}

/** Repository facts for whatever is currently selected, and why they are absent. */
function HfSelectionFacts({ value, subject }: { value: string; subject: string }) {
  const path = isFilesystemPath(value)
  const detail = useHfDetail(value, !path)
  const facts = detail.data && !detail.data.error ? detail.data : undefined

  if (!value) return null

  if (path) {
    return (
      <span className="t-tiny muted">
        That is a path on this machine, so it is not looked up on Hugging Face. The checkpoint's own files supply the
        context window and the architecture when the engine loads it.
      </span>
    )
  }

  return (
    <>
      {facts && (
        <span className="row gap-2 wrap">
          {facts.context_length ? (
            <Badge outline mono>
              {tokens(facts.context_length)} token context
            </Badge>
          ) : null}
          {facts.parameter_count ? (
            <Badge outline mono>
              {compactNumber(facts.parameter_count)} parameters
            </Badge>
          ) : null}
          {facts.model_type && <Badge outline>{facts.model_type}</Badge>}
          {facts.license && <Badge outline>{facts.license}</Badge>}
          {facts.gated && <Badge tone="caution">gated repository</Badge>}
        </span>
      )}
      {facts?.gated && (
        <span className="t-tiny muted">
          A gated repository needs an accepted licence on Hugging Face and a stored token.
        </span>
      )}
      {(detail.isError || detail.data?.error) && (
        <span className="t-tiny muted">
          This repository could not be looked up, so nothing is known about it here. The run will still try to load{' '}
          {subject}.
        </span>
      )}
    </>
  )
}

export function HfModelSearch() {
  const { config, update } = useSettingsController()
  const current = config.local.model_name

  /**
   * The context window is written back only once the lookup returns, several
   * hundred milliseconds during which the selection can move on: the result
   * list stays on screen for a moment after a pick, and the name can be typed
   * over at any time. Two independent conditions retire a pending write, and
   * neither depends on which lookup happens to answer first. The generation
   * retires every earlier pick the instant a new one starts; the name check
   * retires a write whose model is no longer the configured one.
   */
  const generation = useRef(0)
  const chosen = useRef(current)
  chosen.current = current

  const pick = (id: string) => {
    const mine = (generation.current += 1)
    chosen.current = id
    update('local', { model_name: id })
    gatedDetail(id)
      .then((row) => {
        // A later pick retires this one, whichever lookup answers first; a name
        // typed over the top retires it too.
        if (mine !== generation.current || chosen.current !== id) return
        if (row.context_length && row.context_length > 0) update('local', { max_model_len: row.context_length })
      })
      .catch(() => undefined)
  }

  return (
    <div className="stack gap-3">
      <HfSearchList
        label="Search Hugging Face"
        hint="Pick a repository and its context window is written into the configuration below."
        placeholder="qwen3 instruct awq"
        selected={current}
        onPick={pick}
      />

      <TextSetting
        label="Model name"
        section="local"
        field="model_name"
        mono
        placeholder="Qwen/Qwen3-14B-AWQ"
        hint="A Hugging Face repository id, or an absolute path to a checkout already on disk."
      />

      <HfSelectionFacts value={current} subject="it" />
    </div>
  )
}

/**
 * The embedding model, when the weights run here.
 *
 * A self-hosted run does not call a provider for embeddings: `get_embedding`
 * loads this value with sentence-transformers on the CPU, so it has to be a
 * sentence-transformers repository id or a checkout on disk. A provider route
 * would be handed straight to `SentenceTransformer(...)` and fail there.
 */
export function HfEmbeddingSearch() {
  const { config, update } = useSettingsController()
  const current = config.models.embedding_model

  return (
    <div className="stack gap-3">
      <HfSearchList
        label="Search Hugging Face for an embedding model"
        hint="Feature-extraction repositories only. Leaving the field below empty loads BAAI/bge-large-en-v1.5."
        placeholder="bge gte e5"
        task="embedding"
        selected={current}
        onPick={(id) => update('models', { embedding_model: id })}
      />

      <TextSetting
        label="Embedding model"
        section="models"
        field="embedding_model"
        mono
        placeholder="BAAI/bge-large-en-v1.5"
        hint="Deduplicates and clusters evidence before the report is written. It is loaded on this machine with sentence-transformers, on the CPU, so it is a repository id or a path to a checkout on disk rather than a provider route."
      />

      <HfSelectionFacts value={current} subject="the embedding model" />
    </div>
  )
}

/* --- Deployment plan ------------------------------------------------------ */

export function DeploymentPlan({ local }: { local: LocalBackendConfig }) {
  const probe = useDeployProbe()
  const plan = useDeployPlan(local.model_name ? local : null)

  if (!local.model_name) {
    return <span className="t-tiny muted">Choose a model and the launch command appears here.</span>
  }
  if (plan.isLoading) {
    return (
      <div className="row gap-2">
        <Spinner size={13} /> <span className="t-tiny muted">Resolving the plan</span>
      </div>
    )
  }
  if (plan.isError || !plan.data) {
    return (
      <span className="t-tiny muted">
        The deployment plan could not be resolved, so no command is shown. The settings below are still saved.
      </span>
    )
  }

  return <PlanBody plan={plan.data} local={local} probe={probe.data} />
}

/**
 * Reading the plan.
 *
 * The service is reshaping this payload while this screen is in use, so nothing
 * here is destructured on faith. `commands` has arrived both as a flat list
 * tagged with its runtime and as an object of one named group per runtime, and
 * the memory figure has moved from a scalar into a nested object. Both shapes
 * render; anything unrecognised renders as no command at all, which is a worse
 * screen but not a thrown exception that takes the whole dialog with it.
 */
interface PlanCommand {
  key: string
  title: string
  command: string
  note?: string
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null

const asNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null

const asText = (value: unknown): string => (typeof value === 'string' ? value : '')

/** The named commands of one runtime group, in the order the service sent them. */
const groupCommands = (group: Record<string, unknown> | null): PlanCommand[] =>
  group
    ? Object.entries(group)
        .filter(([, value]) => typeof value === 'string' && value.trim().length > 0)
        .map(([key, value]) => ({ key, title: titleCase(key), command: String(value) }))
    : []

function readCommands(
  plan: DeployPlan,
  runtime: LocalRuntime,
): { primary: PlanCommand[]; batch: PlanCommand[] } {
  const raw: unknown = (plan as unknown as Record<string, unknown>).commands

  if (Array.isArray(raw)) {
    const rows = raw.filter((entry): entry is DeployCommand => Boolean(asRecord(entry)) && asText((entry as DeployCommand).command).length > 0)
    const one = (name: string) =>
      rows
        .filter((entry) => entry.runtime === name)
        .map((entry) => ({ key: entry.runtime, title: entry.title, command: entry.command, note: entry.note }))
    return { primary: one(runtime), batch: one('slurm') }
  }

  const selected = groupCommands(asRecord((plan as unknown as Record<string, unknown>).selected_runtime_commands))
  const byRuntime = asRecord(raw)
  return {
    primary: selected.length > 0 ? selected : groupCommands(asRecord(byRuntime?.[runtime])),
    batch: groupCommands(asRecord(byRuntime?.slurm)),
  }
}

function PlanBody({ plan, local, probe }: { plan: DeployPlan; local: LocalBackendConfig; probe?: DeployProbe }) {
  const record = plan as unknown as Record<string, unknown>
  const vram = asRecord(record.vram)
  const needed = asNumber(record.estimated_vram_gb) ?? asNumber(vram?.total_gb)
  const basis = asText(record.estimate_basis) || asText(vram?.basis)
  const available = probe?.total_vram_gb ?? null
  const short = needed !== null && available !== null && needed > available

  // A service that predates the runtime fields sends neither, so both fall back
  // to the plain on-host case rather than rendering an unlabelled control.
  const runtime = local.runtime ?? 'native'
  const { primary, batch } = readCommands(plan, runtime)

  const warnings = Array.isArray(plan.warnings) ? plan.warnings : []
  const notes = Array.isArray(record.notes) ? (record.notes as unknown[]).map(asText).filter(Boolean) : []
  const tensor = asNumber(record.tensor_parallel_size) ?? 1
  const pipeline = asNumber(record.pipeline_parallel_size) ?? 1

  return (
    <div className="stack gap-3">
      <div className="row gap-2 wrap">
        {plan.engine && <Badge tone="accent">{plan.engine}</Badge>}
        <Badge outline mono>
          {asNumber(record.gpu_count) ?? 0} GPU
        </Badge>
        {tensor > 1 && (
          <Badge outline mono>
            TP {tensor}
          </Badge>
        )}
        {pipeline > 1 && (
          <Badge outline mono>
            PP {pipeline}
          </Badge>
        )}
      </div>

      {needed !== null && (
        <span className="t-small" style={short ? { color: 'var(--caution)' } : undefined}>
          Needs about {Math.round(needed)} GB of VRAM
          {available !== null ? `, and this machine reports ${Math.round(available)} GB` : ''}.
          {basis && <span className="t-tiny muted"> {basis}</span>}
        </span>
      )}

      {warnings.length > 0 && (
        <Callout tone="caution" icon={<TriangleAlert size={15} />}>
          <ul className="settings__warnings">
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </Callout>
      )}

      {primary.length > 0 ? (
        primary.map((command) => (
          <CommandBlock key={command.key} title={command.title} command={command.command} note={command.note} />
        ))
      ) : (
        <span className="t-tiny muted">No launch command was returned for the {runtime} runtime.</span>
      )}

      {local.scheduler === 'slurm' &&
        batch.map((command) => (
          <CommandBlock key={command.key} title={command.title} command={command.command} note={command.note} script />
        ))}

      {notes.length > 0 && (
        <ul className="settings__warnings t-tiny muted">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      {asText(record.endpoint_hint) && <span className="t-tiny muted">{asText(record.endpoint_hint)}</span>}
    </div>
  )
}

function CommandBlock({
  title,
  command,
  note,
  script,
}: {
  title: string
  command: string
  note?: string
  script?: boolean
}) {
  return (
    <div className="stack gap-2">
      <div className="row between gap-2">
        <span className="t-tiny semibold">{title}</span>
        <CopyButton text={command} />
      </div>
      <pre className={script ? 'settings__script settings__script--tall' : 'settings__script'}>{command}</pre>
      {note && <span className="t-micro muted">{note}</span>}
    </div>
  )
}

/* --- Runtime controls ----------------------------------------------------- */

export function SelfHostedRuntime() {
  const { config, update } = useSettingsController()
  const local = config.local
  // A service that predates the runtime fields sends neither, so both fall back
  // to the plain on-host case rather than rendering an unlabelled control.
  const runtime = local.runtime ?? 'native'
  const scheduler = local.scheduler ?? 'none'

  return (
    <div className="stack gap-5">
      <Group title="Where it runs">
        <Field label="Runtime" hint={RUNTIME_NOTE[runtime]}>
          <Segmented value={runtime} options={RUNTIMES} onChange={(next) => update('local', { runtime: next })} />
        </Field>
        {runtime !== 'native' && (
          <TextSetting
            label="Image"
            section="local"
            field="image"
            mono
            placeholder={runtime === 'docker' ? 'vllm/vllm-openai:latest' : '/shared/images/vllm.sif'}
            hint={
              runtime === 'docker'
                ? 'A container reference the host can pull.'
                : 'A SIF path, or a docker:// reference Apptainer can convert.'
            }
          />
        )}
        <Field label="Scheduler" hint="SLURM writes an sbatch script instead of starting the engine on this host.">
          <Segmented value={scheduler} options={SCHEDULERS} onChange={(next) => update('local', { scheduler: next })} />
        </Field>
        {scheduler === 'slurm' && (
          <>
            <Grid columns={3}>
              <TextSetting
                label="Partition"
                section="local"
                field="slurm_partition"
                mono
                placeholder="gpu"
                hint="The queue the job is submitted to."
              />
              <TextSetting
                label="Account"
                section="local"
                field="slurm_account"
                mono
                placeholder="Leave empty for your default"
                hint="Charged for the allocation."
              />
              <TextSetting
                label="Wall time"
                section="local"
                field="slurm_time"
                mono
                placeholder="04:00:00"
                hint="HH:MM:SS. The job is killed at this point."
              />
            </Grid>
            <Grid columns={3}>
              <NumberSetting label="Nodes" section="local" field="slurm_nodes" min={1} max={64} hint="One to 64." />
              <NumberSetting
                label="GPUs per node"
                section="local"
                field="slurm_gpus_per_node"
                min={0}
                max={16}
                hint="Requested as gres:gpu."
              />
              <NumberSetting
                label="CPUs per task"
                section="local"
                field="slurm_cpus_per_task"
                min={1}
                max={256}
                hint="Sized for the data loader, not the model."
              />
            </Grid>
            <NumberSetting
              label="Memory"
              section="local"
              field="slurm_mem_gb"
              min={0}
              max={4096}
              step={8}
              width={140}
              suffix="GB"
              hint="Host memory, not VRAM. Zero leaves it to the partition default."
            />
          </>
        )}
      </Group>

      <Group title="Engine">
        <Field label="Engine" hint="vLLM is much faster when it is available; transformers always works.">
          <Select
            value={local.engine}
            onChange={(event) => update('local', { engine: event.target.value as LocalBackendConfig['engine'] })}
          >
            {ENGINES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        </Field>
        <Grid>
          <TextSetting
            label="Dtype"
            section="local"
            field="dtype"
            mono
            placeholder="auto"
            hint="auto, float16, bfloat16, or float32."
          />
          <TextSetting
            label="Quantization"
            section="local"
            field="quantization"
            mono
            placeholder="none"
            hint="awq, gptq, fp8. Empty for an unquantised checkpoint."
          />
          <TextSetting
            label="KV cache dtype"
            section="local"
            field="kv_cache_dtype"
            mono
            placeholder="auto"
            hint="Trades accuracy for cache headroom."
          />
          <TextSetting
            label="Attention implementation"
            section="local"
            field="attn_implementation"
            mono
            placeholder="auto"
            hint="auto, flash_attention_2, sdpa, or eager."
          />
        </Grid>
      </Group>

      <Group title="Hardware">
        <Grid columns={3}>
          <NumberSetting
            label="GPUs to use"
            section="local"
            field="gpu_count"
            min={0}
            max={64}
            hint="Zero lets the engine take every visible device."
          />
          <NumberSetting
            label="Tensor parallel size"
            section="local"
            field="tensor_parallel_size"
            min={1}
            max={16}
            hint="Shards one layer across this many GPUs."
          />
          <NumberSetting
            label="Pipeline parallel size"
            section="local"
            field="pipeline_parallel_size"
            min={1}
            max={16}
            hint="Splits the layers across this many GPUs."
          />
        </Grid>
        <Field
          label={`GPU memory utilization: ${Math.round(local.gpu_memory_utilization * 100)}%`}
          hint="Fraction of each device vLLM may claim for weights and the KV cache."
        >
          <SliderControl
            value={local.gpu_memory_utilization}
            min={0.05}
            max={1}
            step={0.05}
            format={(value) => `${Math.round(value * 100)}%`}
            onCommit={(next) => update('local', { gpu_memory_utilization: next })}
          />
        </Field>
        <NumberSetting
          label="Max tokens per call"
          section="local"
          field="max_tokens"
          min={1024}
          max={1_000_000}
          step={1024}
          width={180}
          hint="Generation ceiling for one call."
        />
      </Group>

      <Group title="Safety">
        <SwitchRow
          label="Enforce eager"
          hint="Skips CUDA graph capture. Slower, but it avoids graph-capture failures on unusual hardware."
          checked={local.enforce_eager}
          onChange={(next) => update('local', { enforce_eager: next })}
        />
        <SwitchRow
          label="Trust remote code"
          hint="Lets a checkpoint execute its own modelling code. Required by many recent architectures."
          info={
            <p>
              This runs Python that ships with the model repository. Leave it on for well-known checkpoints; turn it
              off when you are loading a repository you have not reviewed.
            </p>
          }
          checked={local.trust_remote_code}
          onChange={(next) => update('local', { trust_remote_code: next })}
        />
      </Group>
    </div>
  )
}
