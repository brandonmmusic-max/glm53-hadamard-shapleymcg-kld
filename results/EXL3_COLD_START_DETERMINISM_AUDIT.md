# EXL3 cold-start determinism audit

Date: 2026-09-05

Status: post-hoc, read-only diagnostic

Evidence level: controlled runtime audit; not sealed qualification and not independent reproduction

## Question and claim boundary

This audit asks why the completed EXL3 repeat 1 from the invalidated Tail-V2
`v1a` campaign and EXL3 repeat 1 from the fresh `v1b` campaign have zero of
32 matching raw-logit hashes and a true-decode KLD mean difference of
`+0.0004534678078313516` nats.

The comparison is diagnostic only. The `v1a` EXL3 slot is preserved failure
evidence and is explicitly excluded from `v1b` analysis by the operational
amendment. Each side is one independently started cold process; the 32 windows
inside a process are subsamples, not 32 independent runtime repetitions. Hash
inequality proves that at least one bit differs, not the cause or materiality of
the difference.

No GPU process, service, image, checkpoint, capture, plan, score, or campaign
artifact was changed or launched for this audit.

## Narrow result

- The two EXL3 processes used the same image, in-container command, 170
  environment key/value pairs, model and persistent-cache mounts, four GPU
  UUIDs, driver, TP4/DCP4/EP4 topology, BF16 activation boundary,
  `B12X_MLA_SPARSE`, `nvfp4_ds_mla`, B12X MoE, `max_num_seqs=1`, and seed 0.
- All 32 request JSON files are byte-identical and their execution order is
  identical. Forced token IDs were verified in every response. Response
  payloads differ only in generated `id` and `created` fields.
- All 32 raw-logit hashes differ. Capture metadata differs only in
  `started_unix_ns`, `completed_unix_ns`, and `raw_sha256`.
- The true-decode means are `0.031158372461100103` for `v1a` and
  `0.031611840268931456` for `v1b`: `v1b - v1a =
  +0.0004534678078313516`, or `+1.4553642312270538%` relative to `v1a`.
  Nineteen window deltas are positive and thirteen are negative; their range is
  `[-0.0020188059846597203, +0.008735548351165308]`. A descriptive paired
  t interval is `[-0.0001815951927037147, +0.001088530808366418]`; this was not
  preregistered and is not a qualification interval. Removing the single
  largest positive delta (`conditional-fit-0046`) changes the diagnostic mean
  to `+0.00018630391933670786`.
- One-token-prefill KLD is exactly equal for all 32 windows. The retained
  per-row KLD arrays remain exactly equal through a 19-to-21-row prefix in
  every window, then diverge: first differing row 19 in 26 windows, row 20 in
  four, and row 21 in two. This is evidence against a different static
  checkpoint or configuration affecting row 0. It is consistent with a
  decode-state or scheduling-sensitive divergence, but does not identify its
  mechanism.
- The only `v1a` server error is an `EngineDeadError` during forced shutdown,
  after all 32 capture-complete records. It cannot explain earlier per-row
  divergence.
- A completed within-`v1b` contrast now exists: P8 repeat 1 and repeat 2 are
  32/32 raw-hash identical; all three KLD summaries and all 32 score arrays are
  exactly identical. Stable P8 argues against a universally nondeterministic
  request/capture/scoring path. It does **not** exonerate every path-dependent
  interaction among capture, CUDA graphs, distributed collectives, attention,
  and the EXL3 MoE implementation.

The strongest source-grounded rival is legal arithmetic-order variation in the
EXL3/B12X serving path. Both EXL3 logs prove breakable CUDA graphs, direct
symmetric-memory DCP A2A, and PYNCCL. The cache namespace pins the B12X source
tree `12c426322cc5d239023b57a4bd5ab0e60c4302e0`. In that tree, the W4A16
workspace sets `deterministic_output=False`; B12X's own documentation and
validation state that the production specialization uses atomic output
reduction, that ordered exactness is unsatisfiable when atomics choose summation
order, and that serving with compile/graphs has no bit-reproducibility contract.
Neither launch sets `B12X_DYNAMIC_DETERMINISTIC_OUTPUT`, and the EXL3 launch
does not set `VLLM_EXL3_MCG_A8MX`.

This mechanism fits the observation, but the current evidence does not prove
that it caused the observed difference. Raw logits were retired under the
predeclared storage policy, so exact ULP and layer-boundary localization are not
available from these two runs.

## Evidence matrix

| Surface | `v1a` EXL3 repeat 1 | `v1b` EXL3 repeat 1 | Audit conclusion |
|---|---|---|---|
| Image | `sha256:af4e6a9ac0feb29ae2399ea45fad7db189594f0ff564c5bff3a691156c60b6b7` | same | Exact match |
| Runtime audit | `b5b1c199e7f6af55f86b3a350b29d04d0f0ddf532f098f1a10f598a6b0ac1e61` | same | Byte-identical before and after |
| Container environment | 170 key/value pairs | same 170 pairs | Exact set match; serialized order differs |
| Command/topology | TP4/DCP4/EP4, BF16, EXL3, B12X, NVFP4-DS-MLA | same | Exact match |
| GPU/driver | four matching UUIDs; `610.57.04` | same | Exact identity match |
| Initial idle state | P8 clocks; 31/27/28/29 C | P8 clocks; 45/29/31/36 C | Temperature differs; no evidence that it causes numerical drift |
| Requests | 32 files | same bytes | 32/32 byte-identical |
| Order/concurrency | same frozen order; `max_num_seqs=1` | same | Sequential inputs match |
| HTTP responses | fixed-token content | same content | Only generated `id` and `created` differ |
| Capture metadata | same static schema/shape/token hash | same | Only timestamps and raw hash differ |
| Raw-logit hashes | 32 | 32 | 0/32 equal |
| True-decode KLD mean | `0.031158372461100103` | `0.031611840268931456` | Diagnostic delta `+0.0004534678078313516` |
| Per-row score onset | exact prefix | exact prefix | First difference at row 19/20/21 |
| Startup/runtime markers | graphs, PYNCCL, direct DCP A2A | same markers/counts | Rank log interleaving differs; semantics match |
| Shutdown | forced shutdown error after final capture | clean shutdown | Too late to explain the output difference |

## Immutable identities and receipts

### Plans, images, and activation source

| Object | SHA-256 or immutable identity |
|---|---|
| `v1a` plan | `7917435982fe66426e3f836d96ae90842601e4b801b9b34470f55b41802da135` |
| `v1b` plan | `d055d4db3563fa112326979c70da9ddc058abd5de6f67213aeee3c7308820beb` |
| `v1b` amendment | `6d1cb6e9d9edad5e118fbd14da9df5e7dffb229a9c6ba4e6dd73c70c767e2a11` |
| EXL3 image | `sha256:af4e6a9ac0feb29ae2399ea45fad7db189594f0ff564c5bff3a691156c60b6b7` |
| EXL3 image receipt | `1558102308f101951a2f8fcff5b9103d709960c92dfff4a66c157b62cbdf9d29` |
| Parent image | `sha256:c0f334320c5616392c115279e8a03ac977590ca318cb6f7801fddc0b9c955049` |
| Activation-boundary receipt | `3fe2d7013894c498a00ae58c7e8d92c2356ebc06e69da4a608698be99bcb8583` |
| In-image EXL3 source | `ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45` |
| Patched KPool source | `494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251` |
| Tail-V2 patch source | `b83ba35895a5dfaa2478e3216b4ebc68bb74f73df22931d9ca8ffbdedaf7e7c7` |
| EXL3 product Dockerfile | `899c3abef776416eb125d0965acaf8a6939d10a26ad0ce1cf9a0cf2cdeaf5ef6` |
| EXL3 image builder | `18127237a669790c29162168717186cf4125f0a9b80bc5574d9960c77717188c` |
| Pinned B12X source tree | `12c426322cc5d239023b57a4bd5ab0e60c4302e0` |
| Pinned vLLM source tree | `174c789e09984049d0d53b261024460ca5e9c449` |

The complete source-hash maps are embedded in both sealed plan JSON files.
Only four entries differ between the maps:

| Source | `v1a` SHA-256 | `v1b` SHA-256 |
|---|---|---|
| `glm53_nvfp4/tail_v2_product_runner.py` | `6b67725be05375553dcbaa18a187d96cdc14a7cff4eef97b16a292c04bf2d8ef` | `dd59699d8aeca0ce0c8476ee55f8a8637ca0d9785a559b56215ce1dee56268fd` |
| `glm53_nvfp4/tail_v2_product_validation.py` | `bb06c1c5160b8e05eb074b43211dfde55a54c314955f835561a3bfbe65cfa16f` | `588654c3efb10aabae74ce1628db8359f8651b617dc9b786a33cf431deee8bb4` |
| `scripts/prepare_tail_v2_product_validation.py` | `2cafb59dc6fc67139b72e6b404b4bfb8f7d8cbd2ea7c0f0da6af2e422929afb3` | `e66655e8b838823973c9d5ee6fa66d4962173d305de7e11af4fa3ddc16eb5311` |
| `tests/test_tail_v2_product_validation.py` | `b7495662596d0c4256842d4b3b045a1ed7e5975e6ac7966caa956d5e2da20e9d` | `f3f4b07ae8d01bc7a6ca32abc188129adafdbbf5dee58382c9f57d13b774f7b5` |

Git commit `8cf3d3c27aad42edd3fc03337748ac06977a644c` shows that the runner change is
the non-listening `SO_REUSEADDR` port probe; the other changes authenticate and
record the amended lineage and test the probe. The launch specifications differ
only in the cidfile path, ownership-plan label, and host capture directory.

### Process receipts and logs

| Object | `v1a` SHA-256 | `v1b` SHA-256 |
|---|---|---|
| Launch specification | `c135aa1f7e8630554743c56c1d8a68df6f5a825e451bcde98578144be16fca5f` | `0dd6d8fe2efe286cd3cc7b8798eb40795a8e861ba0f193b987126eb0de09f431` |
| Container inspection | `08117400ffec045c807f10e231a5d8da6047054abb6f832d595f7ec2294d4dc8` | `43c61ca76abfd8e5bc7845d40b284c51930060b3dc721a956a19f4a5ff4e003d` |
| Runtime audit | `b5b1c199e7f6af55f86b3a350b29d04d0f0ddf532f098f1a10f598a6b0ac1e61` | same |
| Execution receipt | `25ad5a902cff14fc1357231e8b3adc92139b931b1afc4e6f5a02d34f92a7a1f2` | `406ce1fa0a00404e049f1da84a93f173d62fc1f835474dbd084f6eb2ffeeffc8` |
| Final server log | `0177a918d2ab8f52b39012eab66afb5ffabbd8c5427eb7ce886ea4f830a9f6f9` | `1910606ae956c65a82927b90b9f6d3d6e0003e6fdc47e7eefc7b5a0b10186e3c` |
| Pre-run NVIDIA XML | `8f7bdd99ca11d0ca390cdabf1099240f880119879bb0ac504a6b85956d7d6795` | `1d72c3a4357ddce726a72e0d5128b1868c1bd95df083380cecd5b259707f800c` |

### Directory-manifest hashes

Each manifest hash below is SHA-256 over sorted lines of the form
`filename`, two spaces, `file_sha256`, newline. Each set contains 32 files.

| Set | `v1a` manifest SHA-256 | `v1b` manifest SHA-256 |
|---|---|---|
| Requests | `d76d3b3cb0749525963da0c368b1afe032c8d72f9e51a304993e0c7c5d25b4b9` | same |
| Responses | `2897755f966917b4ff002992eddde46bfb9c1bbf87048dca38eeb9722575cdd6` | `7fabcc1cca5f4e7e4e308a6551a1d16c4bccc8097ca930c985d2a4d50cf69df1` |
| Capture metadata | `a3dcdbcc46ea731bd11181b053c393d01a6f5ac810e014d9da0722a8ff7cb9b7` | `07c480eabbe6ab2127f537ec4bf6517e6f5ba4424d80a6d0154b0147edf42bda` |
| Score receipts | `d8a01e04adfbb139981b3c0dd1cc354ae52042d2234d5377cefec4c627e1dcde` | `b0a30ed197ea9dc77d51f23049cfc4999fa0fadeea170abf68179d73be36966b` |
| Score arrays | `66913d5f4c7ec9c1d58b214c7ae1dacffbe8362e58b717edae11a7721a973c94` | `1188423eaad4c04a7e3af4454ea2b2b42cfa12116e8f2156410b7ddb6b7e40c9` |
| Retirement receipts | `a8ef2fedd2a5ca403918055effc44fc7996142a3badf30f8214f917bbf51cc15` | `064521c442ed74a8cf688ab2cb1d6cbb7053e0d85f436b48667511d6f97d8d51` |

For the completed P8 within-`v1b` contrast:

| Object | Repeat 1 | Repeat 2 |
|---|---|---|
| Runtime audit | `864e947bc60c0302ebde4ce0dbde55118aebddda0a08050adfe2c6bf07bfe1ab` | same |
| Execution receipt | `6bdc7a314ffc09a45a5a742aeb209d2893288e70a9ac5b0c458d533865c8abaa` | `aa9d7fe6227f0104a7cbd23f69ce4ac389489869018cba4a1569e6b53c03821e` |
| Score-array manifest | `ea960941582d1d505fc32cf8941aaa6cc46551afdb911eee3f2d6a1e00b90e27` | same |
| Raw-hash equality | reference | 32/32 |

Score-receipt manifests differ because repeat metadata differs even though the
P8 score arrays and numerical fields are identical.

## Exact read-only replay commands

Run from the repository root. These commands read existing receipts only.

### Static identity, requests, responses, and capture metadata

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

base = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6')
a = base / 'tail-v2-p8-exl3-cf32-product-validation-v1a/quality/round-01-exl3'
b = base / 'tail-v2-p8-exl3-cf32-product-validation-v1b/quality/round-01-exl3'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def differences(left, right, prefix=''):
    rows = []
    if type(left) is not type(right):
        return [(prefix, left, right)]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                rows.append((f'{prefix}/{key}', left.get(key), right.get(key)))
            else:
                rows.extend(differences(left[key], right[key], f'{prefix}/{key}'))
    elif isinstance(left, list):
        if len(left) != len(right):
            rows.append((f'{prefix}/length', len(left), len(right)))
        else:
            for index, (lv, rv) in enumerate(zip(left, right)):
                rows.extend(differences(lv, rv, f'{prefix}/{index}'))
    elif left != right:
        rows.append((prefix, left, right))
    return rows

ca = json.loads((a / 'container.private.json').read_text())[0]
cb = json.loads((b / 'container.private.json').read_text())[0]
env_a = dict(item.split('=', 1) for item in ca['Config']['Env'] if '=' in item)
env_b = dict(item.split('=', 1) for item in cb['Config']['Env'] if '=' in item)
print('runtime audits', sha(a / 'runtime-audit.json'), sha(b / 'runtime-audit.json'))
print('environment maps equal', env_a == env_b, len(env_a), len(env_b))
for key in ('Image', 'Cmd', 'Entrypoint', 'WorkingDir', 'User'):
    print('container config', key, ca['Config'].get(key) == cb['Config'].get(key))

for kind in ('request', 'response'):
    pa = {p.name: p for p in (a / 'requests').glob(f'*.{kind}.json')}
    pb = {p.name: p for p in (b / 'requests').glob(f'*.{kind}.json')}
    print(kind, 'byte-equal', sum(sha(pa[name]) == sha(pb[name]) for name in pa), '/', len(pa))
    if kind == 'response':
        paths = {}
        for name in sorted(pa):
            for path, _, _ in differences(json.loads(pa[name].read_text()),
                                          json.loads(pb[name].read_text())):
                paths[path] = paths.get(path, 0) + 1
        print('response differing paths', paths)

capture_paths = {}
raw_equal = 0
for pa in sorted((a / 'captures').glob('*.capture.json')):
    pb = b / 'captures' / pa.name
    ja = json.loads(pa.read_text())
    jb = json.loads(pb.read_text())
    raw_equal += ja['raw_sha256'] == jb['raw_sha256']
    for path, _, _ in differences(ja, jb):
        capture_paths[path] = capture_paths.get(path, 0) + 1
print('capture raw hashes equal', raw_equal, '/ 32')
print('capture metadata differing paths', capture_paths)

ea = json.loads((a / 'execution.json').read_text())
eb = json.loads((b / 'execution.json').read_text())
print('window order equal', [row['id'] for row in ea['windows']] ==
      [row['id'] for row in eb['windows']])
PY
```

Expected decisive lines:

```text
runtime audits b5b1c199e7f6af55f86b3a350b29d04d0f0ddf532f098f1a10f598a6b0ac1e61 b5b1c199e7f6af55f86b3a350b29d04d0f0ddf532f098f1a10f598a6b0ac1e61
environment maps equal True 170 170
request byte-equal 32 / 32
response differing paths {'/created': 32, '/id': 32}
capture raw hashes equal 0 / 32
capture metadata differing paths {'/completed_unix_ns': 32, '/raw_sha256': 32, '/started_unix_ns': 32}
window order equal True
```

### Exact KLD and score-array divergence

```bash
python3 - <<'PY'
import collections
import json
import math
from pathlib import Path
import statistics

import numpy as np

base = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6')
sa = base / 'tail-v2-p8-exl3-cf32-product-validation-v1a/quality/scores/repeat-01/exl3'
sb = base / 'tail-v2-p8-exl3-cf32-product-validation-v1b/quality/scores/repeat-01/exl3'
a = {p.name: json.loads(p.read_text()) for p in sa.glob('*.score.json')}
b = {p.name: json.loads(p.read_text()) for p in sb.glob('*.score.json')}

for key in ('mean_kld', 'one_token_prefill_kld', 'true_decode_mean_kld'):
    pairs = [(a[name][key], b[name][key]) for name in sorted(a)]
    delta = [right - left for left, right in pairs]
    print(key,
          'mean_a', statistics.fmean(left for left, _ in pairs),
          'mean_b', statistics.fmean(right for _, right in pairs),
          'mean_delta', statistics.fmean(delta),
          'range', (min(delta), max(delta)),
          'signs', (sum(x > 0 for x in delta), sum(x < 0 for x in delta)))

delta = [b[name]['true_decode_mean_kld'] - a[name]['true_decode_mean_kld']
         for name in sorted(a)]
se = statistics.stdev(delta) / math.sqrt(len(delta))
print('descriptive t interval', statistics.fmean(delta) - 2.04 * se,
      statistics.fmean(delta) + 2.04 * se)

first_difference = []
equal_prefix_rows = []
for pa in sorted(sa.glob('*.scores.npz')):
    with np.load(pa) as za, np.load(sb / pa.name) as zb:
        unequal = np.flatnonzero(za['kld'] != zb['kld'])
        first_difference.append(int(unequal[0]) if len(unequal) else None)
        equal_prefix_rows.append(int(unequal[0]) if len(unequal) else len(za['kld']))
print('first differing row counts', collections.Counter(first_difference))
print('exact prefix range', min(equal_prefix_rows), max(equal_prefix_rows))
PY
```

Expected decisive lines:

```text
one_token_prefill_kld mean_a 3.8809088984823252 mean_b 3.8809088984823252 mean_delta 0.0 range (0.0, 0.0) signs (0, 0)
true_decode_mean_kld mean_a 0.031158372461100103 mean_b 0.031611840268931456 mean_delta 0.0004534678078313516 range (-0.0020188059846597203, 0.008735548351165308) signs (19, 13)
descriptive t interval -0.0001815951927037147 0.001088530808366418
first differing row counts Counter({19: 26, 20: 4, 21: 2})
exact prefix range 19 21
```

### Completed P8 repeat control

```bash
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

base = Path('/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/tail-v2-p8-exl3-cf32-product-validation-v1b/quality/scores')
ra = base / 'repeat-01/p8'
rb = base / 'repeat-02/p8'
a = {p.name: json.loads(p.read_text()) for p in ra.glob('*.score.json')}
b = {p.name: json.loads(p.read_text()) for p in rb.glob('*.score.json')}
print('raw hash equal', sum(a[name]['raw_sha256'] == b[name]['raw_sha256'] for name in a), '/', len(a))
for key in ('mean_kld', 'one_token_prefill_kld', 'true_decode_mean_kld'):
    print(key, all(a[name][key] == b[name][key] for name in a))
print('score arrays byte-equal',
      sum((ra / p.name).read_bytes() == (rb / p.name).read_bytes()
          for p in ra.glob('*.scores.npz')), '/ 32')
PY
```

Expected result: `32 / 32` raw-hash equality, all three numerical checks
`True`, and `32 / 32` byte-identical score arrays.

## Next minimal diagnostic

First complete the already scheduled `v1b` cold-process repeats and build a
five-process-by-32-window hash and KLD matrix separately for EXL3 and P8. This
uses no additional GPU work and is the correct independent-run boundary.

If P8 remains stable and EXL3 does not, do not run another full CF32 campaign.
Predeclare one fixed conditional-fit window before inspecting extremes, retain
the raw logits from two cold EXL3 runs, and record layer-boundary hashes around
decode rows 18 through 22. Then test one single-variable eager/no-CUDA-graph
control. If eager execution still diverges, an ordered-combine W4A16 scoring
implementation is the decisive follow-up: the pinned W4A16 B12X plan hardcodes
`deterministic_output=False`, so merely exporting the general deterministic
environment flag is not sufficient evidence that W4A16 used ordered combine.

Until that diagnostic exists, the supported conclusion is limited to:

> Two otherwise matched EXL3 cold starts produced different decode logits with
> a small, non-systematic aggregate KLD drift beginning after an exact early
> decode prefix. A completed P8 pair is bitwise stable. The evidence localizes
> concern toward an EXL3/path-dependent runtime interaction and identifies
> atomic/graph arithmetic ordering as a credible source-grounded rival, but does
> not establish causality.
