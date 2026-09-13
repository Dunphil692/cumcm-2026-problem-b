"""渲染论文公式为 PNG（matplotlib mathtext），供 Word 文档嵌入。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq")
os.makedirs(OUT, exist_ok=True)

FORMULAS = [
    ("eq1", r"$\Omega_1=\left\lbrace\,g:\ \Vert g\Vert\leq 1800,\ \ 5<\Vert g-S_1\Vert\leq 1500,\ \ \left\Vert \mathrm{wrap}\left(\theta(g-S_1)-\theta_1\right)\right\Vert\leq 1^{\circ}\,\right\rbrace$"),
    ("eq2", r"$R\geq \Vert G-S_1\Vert,\qquad d_2=\Vert S_2-G\Vert\ \leq\ L(G):=\max\left(1000,\ \Vert G-S_1\Vert\right)$"),
    ("eq3", r"$C_{\mathrm{recv}}=\left\lbrace\,q:\ F(q)=\sup_{g\in\Omega_1}\left(\Vert q-g\Vert-L(g)\right)\leq 0\,\right\rbrace,\qquad L(g)=\max(1000,\ \Vert g-S_1\Vert)$"),
    ("eq4", r"$\Omega_2=W_1\,\cap\,W_2\,\cap\,\Omega_1\,\cap\,\left\lbrace\,5<\Vert z-q\Vert\leq 1500\,\right\rbrace$"),
    ("eq5", r"$J(q)=\sup_{g\in\Omega_1}\ \sup_{e_2\in[-1^{\circ},1^{\circ}]}\ \mathrm{diam}\left(\Omega_2(q,g,e_2)\right)$"),
    ("eq6", r"$\mathcal{R}_{\alpha}=\left\lbrace\,q\in C_{\mathrm{recv}}:\ J(q)\leq \alpha\,J_{\min}\,\right\rbrace$"),
    ("eq7", r"$D\ \approx\ \frac{2\tan 1^{\circ}}{\sin\psi}\sqrt{r_1^{\,2}+r_2^{\,2}+2\,r_1 r_2\left|\cos\psi\right|}$"),
    ("eq8", r"$C_{\mathrm{recv}}\supseteq B(S_1,1000)\ \cap\ B(P_-,1000)\ \cap\ B(P_+,1000)$"),
]

failed = []
for name, tex in FORMULAS:
    try:
        fig = plt.figure(figsize=(0.13 * len(tex), 0.7), dpi=300)
        fig.text(0.01, 0.5, tex, fontsize=13, va="center")
        fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"{name}: OK")
    except Exception as e:
        failed.append((name, str(e)))
        print(f"{name}: FAIL -> {e}")

print("failed:", failed)
