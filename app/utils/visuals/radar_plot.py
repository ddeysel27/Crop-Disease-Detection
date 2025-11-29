def radar_chart(conf, mc_unc, tta_unc, stability):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["Confidence", "MC Unc", "TTA Unc", "Stability"]
    values = [conf, 1-mc_unc, 1-tta_unc, 1-stability]  # invert if needed

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    return fig
