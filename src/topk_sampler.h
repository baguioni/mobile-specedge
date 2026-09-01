#pragma once

#include <cstdint>

#include "llama.h"

namespace specedge {

// A llama.cpp *backend* sampler: full-vocabulary softmax followed by top-k,
// both emitted as graph nodes so they execute on whatever device the model
// runs on instead of on the host.
//
// Why this exists rather than llama_sampler_init_top_k(): the stock top-k
// sampler returns the top-k *logits*, and SpecExec needs log-probabilities
// under the full-vocabulary normalizer. A beam's score is the cumulative
// log-probability of its whole root-to-leaf path, and every beam has its
// own normalizer, so normalizing over just the k survivors would silently
// change how beams compare against each other -- exactly the comparison
// that decides the tree's shape. Doing the softmax before the top-k keeps
// the arithmetic identical to the Python reference's
// log_softmax(...).topk(...).
//
// Attaching one of these to every sequence has a second, larger effect:
// llama_context::decode skips its raw-logits device->host copy when every
// output sequence has a backend sampler (see needs_raw_logits() in
// llama-context.cpp). With a 151,936-entry vocabulary and 32 beams that is
// ~19 MB per tree level that never crosses the bus.
//
// Returns a sampler that yields, per output:
//   llama_get_sampled_candidates_ith() -> k token ids
//   llama_get_sampled_probs_ith()      -> their full-vocab probabilities
//   llama_get_sampled_logits_ith()     -> their raw logits (for parity checks)
// The result is a one-entry llama_sampler_chain, because
// llama_context_params::samplers rejects anything else. The caller owns it
// and must llama_sampler_free() it *after* the llama_context that uses it
// is destroyed; freeing the chain frees the sampler inside.
//
// NOTE: llama.cpp marks the backend sampling API [EXPERIMENTAL].
llama_sampler* make_topk_logprob_sampler(int32_t k);

} // namespace specedge
