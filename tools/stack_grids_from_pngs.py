from PIL import Image
import matplotlib.pyplot as plt


def stack_three_grid_pngs(p1, p2, p3, out_path="e5_E1_E2_E5_grids.png"):
    """Takes three images and displays them (might easily be adapted to do any number)"""
    grids = [Image.open(p) for p in [p1, p2, p3]]

    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    titles = ["E1 — Linear β", "E2 — Cosine β", "E5 — Cosine (Σβ matched)"]

    for ax, title, img in zip(axes, titles, grids):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    stack_three_grid_pngs(
        "path/to/E1/grid.png",
        "path/to/E2/grid.png",
        "path/to/E5/grid.png",
    )