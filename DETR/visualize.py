
import matplotlib.pyplot as plt



# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
          

# Plotting results
def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    patch_id = 0
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                                 fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'patch_{patch_id}: {p[cl]:0.2f}'
        patch_id += 1
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

# Visualize patches
def show_patches(patches):
    fig = plt.figure(figsize=(16,len(patches))) 
    for i,patch in enumerate(patches):
        fig.add_subplot(1, len(patches), i+1)
        plt.imshow(patch) 
        plt.axis('off') 
        plt.title(f"patch_{i}") 
    plt.show()
