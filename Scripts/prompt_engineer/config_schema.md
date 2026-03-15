# Prompt Engineer Config Schema

Configuration file: `prompt-engineer.yaml` (project root or specified path)

---

## Top-Level Keys

```yaml
project_name: string        # Human-readable project name
data: object                # Data source configuration
fields: list[FieldDef]      # Classification fields to optimize
models: list[ModelDef]      # LLM models available for testing
budget: BudgetDef           # Spending limits
convergence: ConvergenceDef # When to stop iterating
architecture: ArchDef       # Architecture search options
output_dir: string          # Directory for results, reports, state (default: prompt_engineer_output)
```

---

## `data`

```yaml
data:
  # Path to CSV/JSON with input texts and gold labels
  inputs_path: string           # Required. Path to file with input data
  inputs_format: csv | json     # Default: inferred from extension

  # Column/key mappings
  id_column: string             # Column name for unique ID (default: "id")
  text_column: string           # Column name for input text (default: "transcript")

  # Gold labels source (same file or separate)
  labels_path: string           # Optional. If omitted, labels are in inputs_path
  labels_format: csv | json     # Default: inferred from extension
  labels_id_column: string      # ID column in labels file (default: same as id_column)

  # Sampling
  holdout_ratio: float          # Fraction reserved for final validation (default: 0.2)
  holdout_seed: int             # Random seed for holdout split (default: 42)
```

---

## `fields` (list)

Each field is a classification target.

```yaml
fields:
  - name: string              # Field name (used in output keys)
    type: categorical | text | boolean  # Field type (default: categorical)

    # For categorical fields
    values: list[string]       # Valid output values (enum)
    hierarchy:                 # Optional parent/sub-category structure
      parents:                 # Map of parent -> list of sub-categories
        "Parent A": ["Sub A1", "Sub A2"]
        "Parent B": ["Sub B1"]

    # Dependencies
    depends_on: string         # Another field name this depends on
    skip_when:                 # Skip LLM call when dependency has these values
      field: string            # The dependency field name
      values: list[string]     # Values that trigger skip
      default: any             # Value to assign when skipped (default: null)

    # Evaluation
    gold_column: string        # Column name in labels (default: same as name)
    skip_null_gold: bool       # Skip rows where gold label is null (default: false)
    skip_null_both: bool       # Skip rows where both gold and pred are null (default: false)
    target_accuracy: float     # Target accuracy percentage (e.g., 90.0)

    # Prompt
    system_prompt: string      # Initial system prompt for this field
    response_schema: object    # JSON Schema for structured output (optional)

    # Processing order (lower = earlier, fields at same priority run in parallel)
    priority: int              # Default: 0
```

---

## `models` (list)

```yaml
models:
  - provider: string          # Provider name (gemini, openai, anthropic, or custom)
    name: string              # Model name/ID (e.g., "gemini-2.5-flash")
    tier: cheap | mid | expensive  # Cost tier for architecture search
    api_key_env: string       # Env var name for API key
    base_url: string          # API base URL (optional, uses provider default)
    base_url_env: string      # Env var for base URL (optional)
    max_concurrent: int       # Max concurrent requests (default: 15)
    supports_structured: bool # Supports JSON schema response_format (default: true)

    # Pricing (per million tokens)
    input_price: float        # $/1M input tokens
    output_price: float       # $/1M output tokens

    # Batching
    default_batch_size: int   # Default batch size (default: 1)
    max_batch_size: int       # Max batch size to test (default: 20)

    # Options
    temperature: float        # Default: 0.0
    supports_temperature: bool # Some models ignore temperature (default: true)
```

---

## `budget`

```yaml
budget:
  max_total_usd: float       # Hard stop on total spend (default: 10.0)
  warn_at_usd: float         # Log warning at this threshold (default: 5.0)
  max_per_phase_usd: float   # Per-phase limit (default: 3.0)
  track_file: string         # Cost tracking file (default: "{output_dir}/costs.json")
```

---

## `convergence`

```yaml
convergence:
  min_rounds: int             # Minimum OPRO rounds before checking (default: 3)
  max_rounds: int             # Maximum OPRO rounds total (default: 15)
  plateau_rounds: int         # Stop if < threshold gain for N consecutive rounds (default: 3)
  plateau_threshold: float    # Minimum gain in pp to count as improvement (default: 1.0)
  sample_sizes:               # Sample sizes for different phases
    baseline: int             # Phase 1 sample size (default: 50)
    iteration: int            # Phase 4 OPRO iteration size (default: 100)
    confirmation: int         # Confirm gains at larger N (default: 200)
    final: int                # Phase 6 final validation (default: full holdout)
```

---

## `architecture`

```yaml
architecture:
  test_batch_sizes: list[int]   # Batch sizes to compare (default: [1, 5, 10, 15])
  test_models: bool             # Compare models across tiers (default: true)
  test_decomposition: bool      # Test field decomposition strategies (default: true)
  decomposition_options:
    - single_call              # All fields in one prompt
    - per_field                # Separate prompt per field
    - two_step                 # Reasoning step + classification step
```

---

## `cross_field_rules` (list, optional)

Post-classification consistency rules applied during assembly.

```yaml
cross_field_rules:
  - name: string              # Rule name for logging
    condition:                # When to trigger
      field: string           # Field to check
      value: any              # Value that triggers rule
    action:                   # What to do
      set_field: string       # Field to modify
      to_value: any           # Value to set
    severity: error | warning # Log level (default: warning)
```

---

## `flags` (optional)

Input quality flagging configuration.

```yaml
flags:
  keyword_patterns:           # Regex patterns for domain-specific detection
    medical: "sick|vomit|limp|..."
    admin: "reschedule|cancel|..."
  rules:
    - name: string            # Flag name
      condition: string       # Python expression evaluated per input
      # Available variables: turns, caller_words, total_words, text_length, keyword_matches
```

---

## Full Example

See `example_config.yaml` for a complete working configuration based on the CCR Dashboard veterinary call classification project.
