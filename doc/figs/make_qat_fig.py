"""
make_qat_fig.py — fig-qat.png of the "QAT" section of ../nnue_roadmap.qmd:
(left) what the forward pass and the gradient see of the rounding of a weight;
(right) the value of spacelite on 2000 positions, the float network (what it
was trained as) against the int8 network the C++ runs.

qat_values.csv: the float values (check_persp_net.py on spacelite) and the C++
ones (check_persp_net.py on spacelite with qat_t = 1, which equals the C++
bit for bit: nets_check check_persp, max |err| 0).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.1))
    w = np.linspace(-0.07, 0.07, 2001)
    a.plot(w * 64, np.round(w * 64), color="#246", lw=2, label="passage avant : round(64 w)")
    a.plot(w * 64, w * 64, color="#d62", lw=1.5, ls="--", label="gradient : comme si c'était l'identité")
    a.set_xlabel("64 × poids float (maître)")
    a.set_ylabel("poids int8 utilisé")
    a.set_title("l'estimateur « straight-through »")
    a.legend(frameon=False, fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    d = np.loadtxt(os.path.join(HERE, "qat_values.csv"), delimiter=",", skiprows=1)
    f, q = d[:, 0], d[:, 1]
    e = np.abs(f - q)
    b.scatter(f, q, s=4, alpha=0.4, color="#246")
    b.plot([-1, 1], [-1, 1], color="k", lw=0.6)
    big = e > 0.1
    b.scatter(f[big], q[big], s=10, color="#d62", label=f"|écart| > 0,1 : {big.mean() * 100:.1f} % des positions".replace(".", ","))
    b.set_xlabel("valeur du réseau float (entraîné)")
    b.set_ylabel("valeur du réseau int8 (C++)")
    b.set_title(f"spacelite, 2000 positions : écart moyen {e.mean():.3f}".replace(".", ","))
    b.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig-qat.png"), dpi=200)


if __name__ == "__main__":
    main()
