"""
Fid heatmap plot fun thing.

currently only takes manually inputed numbers:

Current:
    python -m tools.plot_e6_heatmap \
    --fid-linear-ddpm 193.1787 \
    --fid-linear-ddim 194.9572 \
    --fid-cosine-ddpm 194.2269 \
    --fid-cosine-ddim 194.1266 \
    --out-plot docs/assets/E6/e6_plots/e6_fid_heatmap.png

"""


import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Really fast orchestrator:"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid-linear-ddpm", type=float, required=True)
    parser.add_argument("--fid-linear-ddim", type=float, required=True)
    parser.add_argument("--fid-cosine-ddpm", type=float, required=True)
    parser.add_argument("--fid-cosine-ddim", type=float, required=True)
    parser.add_argument("--out-plot", type=str, required=True)
    args = parser.parse_args()

    # 2x2 matrix: rows = schedule, cols = sampler
    data = np.array([
        [args.fid_linear_ddpm, args.fid_linear_ddim],   # linear row
        [args.fid_cosine_ddpm, args.fid_cosine_ddim],   # cosine row
    ])

    row_labels = ["linear", "cosine"]
    col_labels = ["DDPM", "DDIM"]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, aspect="equal")

    # Tick labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Move x labels to top if you like
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Annotate each cell with its value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(
                j, i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if val > data.mean() else "black",
            )

    # Colorbar for scale
    cbar = plt.colorbar(
        im,
        ax=ax,
        shrink=0.7,    # shorter bar (0.7 = 70% of default length)
        fraction=0.046,  # thinner bar relative to the axis size
        pad=0.04       # space between heatmap and colorbar
    )
    cbar.set_label("FID", rotation=270, labelpad=8)

    ax.set_title("E6: FID heatmap (schedule x sampler)!")
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=200)
    print(f"Saved heatmap to {args.out_plot}")

if __name__ == "__main__":
    main()


