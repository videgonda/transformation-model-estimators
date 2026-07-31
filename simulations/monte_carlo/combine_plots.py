import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def merge_plots(files, output_path):

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    for ax, img_path in zip(axes, files):
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
        else:
            print(f"Warning: Image not found at {img_path}")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    folder = os.path.dirname(__file__)

    merge_plots([os.path.join(folder, "chen_design_1.png"), os.path.join(folder, "chen_design_2.png"), os.path.join(folder, "chen_design_3.png")], os.path.join(folder, "chen_merged.png"))
    merge_plots([os.path.join(folder, "ye_duan_design_1.png"), os.path.join(folder, "ye_duan_design_2.png"), os.path.join(folder, "ye_duan_design_3.png")], os.path.join(folder, "ye_duan_merged.png"))
    merge_plots([os.path.join(folder, "engression_design_1.png"), os.path.join(folder, "engression_design_2.png"), os.path.join(folder, "engression_design_3.png")], os.path.join(folder, "engression_merged.png"))




if __name__ == "__main__":
    main()