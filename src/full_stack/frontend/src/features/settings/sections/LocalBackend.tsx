/** Everything the on-host inference server needs when the backend is Local. */

import { Callout, Field, Select } from '@/components/ui/primitives'
import { Grid, Group, NumberSetting, SectionHead, SliderControl, SwitchRow, TextSetting } from '../controls'
import { useSettingsController } from '../state'

const ENGINES: { value: 'auto' | 'vllm' | 'transformers'; label: string }[] = [
  { value: 'auto', label: 'Auto: vLLM when it imports, transformers otherwise' },
  { value: 'vllm', label: 'vLLM' },
  { value: 'transformers', label: 'Transformers' },
]

export function LocalSection() {
  const { config, update } = useSettingsController()
  const local = config.local
  const active = config.connection.backend === 'local'

  return (
    <>
      <SectionHead
        title="Local backend"
        description="Model and runtime for inference on this machine."
      />

      {!active && (
        <Callout tone="neutral">
          These settings only apply when the backend is set to Local. The backend is currently{' '}
          <span className="semibold">{config.connection.backend}</span>, so nothing here affects a run yet.
        </Callout>
      )}

      <Group title="Model">
        <TextSetting
          label="Model name"
          section="local"
          field="model_name"
          mono
          placeholder="Qwen/Qwen3-14B-AWQ"
          hint="A Hugging Face repository id, or an absolute path to a local checkout."
        />
        <Grid>
          <NumberSetting
            label="Max tokens"
            section="local"
            field="max_tokens"
            min={1024}
            max={1_000_000}
            step={1024}
            hint="Generation ceiling per call. At least 1024."
          />
          <NumberSetting
            label="Max model length"
            section="local"
            field="max_model_len"
            min={0}
            max={4_000_000}
            step={1024}
            hint="Zero uses the context length declared by the checkpoint."
          />
        </Grid>
      </Group>

      <Group title="Runtime">
        <Field label="Engine" hint="vLLM is much faster when it is available; transformers always works.">
          <Select
            value={local.engine}
            onChange={(event) => update('local', { engine: event.target.value as typeof local.engine })}
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
            hint="awq, gptq, fp8, and so on. Leave empty for an unquantised checkpoint."
          />
          <TextSetting
            label="KV cache dtype"
            section="local"
            field="kv_cache_dtype"
            mono
            placeholder="auto"
            hint="auto, fp8, or fp8_e5m2. Trades accuracy for cache headroom."
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
        <Grid>
          <NumberSetting
            label="Tensor parallel size"
            section="local"
            field="tensor_parallel_size"
            min={1}
            max={16}
            hint="Shards one layer across this many GPUs. One to sixteen."
          />
          <NumberSetting
            label="Pipeline parallel size"
            section="local"
            field="pipeline_parallel_size"
            min={1}
            max={16}
            hint="Splits the layers across this many GPUs. One to sixteen."
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
            presets={[0.5, 0.7, 0.9, 0.95]}
            onCommit={(next) => update('local', { gpu_memory_utilization: next })}
          />
        </Field>
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
              This runs Python that ships with the model repository. Leave it on for well-known checkpoints; turn it off
              when you are loading a repository you have not reviewed.
            </p>
          }
          checked={local.trust_remote_code}
          onChange={(next) => update('local', { trust_remote_code: next })}
        />
      </Group>
    </>
  )
}
