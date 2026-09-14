"""Recompute round time with the network held fixed.

client_wait = network round trip + server compute. The server logs its own
end_to_end per round (std < 1 ms), so subtracting it leaves the transport, and
substituting a chosen transport gives what the run would have looked like on a
different link.

The proactive draft runs *inside* that window, so the phase is not additive:
it takes max(network + server, proactive_ms). Shrinking the network therefore
also shrinks the free overlap, which is the honest way to model it -- a
co-located target leaves proactive draft nowhere to hide.
"""
import json
import statistics as st
import sys

RUNS = ("baseline-cpu", "proactive-cpu", "baseline-gpu", "proactive-gpu")


def load(name):
    cl = [r for r in (json.loads(l) for l in open(f"result/phone/{name}/client_0.jsonl") if l.strip())
          if not r["target"].get("prefill")]
    srv = [json.loads(l) for l in open(f"result/phone/{name}/server.jsonl") if l.strip()]
    server_ms = st.mean([s["target"]["server_end_to_end_t"] for s in srv if not s["target"].get("prefill")])
    return cl, server_ms


def round_ms(r, server_ms, network_ms):
    t = r["target"]
    window = max(network_ms + server_ms, t.get("proactive_ms", 0.0) or 0.0)
    return r["draft"]["end_to_end"] + t["client_preprocess"] + t["client_postprocess"] + window


def main():
    networks = [("observed", None), ("278 ms (best observed)", 278.0), ("0 ms (co-located)", 0.0)]
    print(f"{'run':<16}" + "".join(f"{lbl:>26}" for lbl, _ in networks))
    print("-" * (16 + 26 * len(networks)))
    for name in RUNS:
        cl, server_ms = load(name)
        acc = sum(r["num_accepted_tokens"] for r in cl)
        cells = []
        for _, net in networks:
            if net is None:
                tot = sum(r["draft"]["end_to_end"] + r["target"]["end_to_end"] for r in cl)
            else:
                tot = sum(round_ms(r, server_ms, net) for r in cl)
            cells.append(f"{acc/(tot/1000):.2f} tok/s  {tot/acc:6.1f} ms/tok")
        print(f"{name:<16}" + "".join(f"{c:>26}" for c in cells))

    print()
    print(f"{'run':<16}{'draft':>10}{'pre+post':>10}{'network':>10}{'server':>9}{'proactive':>11}")
    print("-" * 66)
    for name in RUNS:
        cl, server_ms = load(name)
        net = st.mean([r["target"]["client_wait"] for r in cl]) - server_ms
        pp = st.mean([r["target"]["client_preprocess"] + r["target"]["client_postprocess"] for r in cl])
        pro = [r["target"]["proactive_ms"] for r in cl if r["target"].get("proactive_ran")]
        print(f"{name:<16}{st.mean([r['draft']['end_to_end'] for r in cl]):>9.1f}{pp:>10.1f}"
              f"{net:>10.1f}{server_ms:>9.1f}{(st.mean(pro) if pro else 0):>11.1f}")


if __name__ == "__main__":
    main()
