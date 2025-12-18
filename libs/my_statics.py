import scipy.stats as stats

def ttest(data1, data2, *args, **kwargs):
    statics, p_value = stats.ttest_ind(data1, data2, *args, **kwargs)

    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'  # Not Significant (有意差なし)
    
    return statics, p_value, sig_text

def violin(data, ax, positions, color):
    parts = ax.violinplot(data, positions=[positions], showmeans=True)
    for body in parts['bodies']:
        body.set_facecolor(color) # 色を指定
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor(color)
        vp.set_linewidth(1.5)