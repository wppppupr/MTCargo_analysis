import zarr
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def get_experiment_rdfs(base_path):
    """
    指定されたパスパターンにマッチする各ディレクトリのRDFを読み込み、
    実験（ディレクトリ）ごとの平均RDFのリストを返す。
    """
    dirs = glob.glob(str(base_path))
    exp_means = []
    
    for d in dirs:
        zarr_path = os.path.join(d, "RDF.zarr")
        # RDF.zarr が存在するものだけをロード
        if os.path.exists(zarr_path):
            print(f"Loading: {zarr_path}")
            rdf = zarr.open_array(zarr_path, mode='r')[:]
            if len(rdf) > 0:
                # このディレクトリ内の全フレーム・粒子の平均を計算し、1つの実験データとする
                exp_means.append(np.nanmean(rdf, axis=0))
                
    return np.array(exp_means)

def plot_RDF(base_path, ax=None, diameter=0.63, color='C0', marker="^", scale=0.11):
    exp_means = get_experiment_rdfs(base_path)
    
    if len(exp_means) > 0:
        # x軸（距離）の計算
        max_r = exp_means.shape[1]
        r = np.arange(max_r) * scale
        
        # 実験間の平均と標準誤差を計算 (N = 実験数)
        rdf_mean = np.nanmean(exp_means, axis=0)
        rdf_sem = np.nanstd(exp_means, axis=0) / np.sqrt(len(exp_means))
        
        # グラフの描画
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots()
            show_plot = True
            
        ax.plot(r, rdf_mean, label=f'{diameter} μm', color=color, marker=marker)
        ax.fill_between(r, rdf_mean - rdf_sem, rdf_mean + rdf_sem, color=color, alpha=0.3)
        
        # バルク平均（g(r)=1）の基準線
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel('Distance [μm]')
        ax.set_ylabel('RDF $g(r)$')
        ax.set_title('Radial Distribution Function')
        ax.legend()

        if show_plot:
            plt.show()
    else:
        print(f"有効な RDF.zarr が見つかりませんでした: {base_path}")

def plot_potential(base_path, ax=None, diameter=0.63, color='C0', marker="^", scale=0.11):
    exp_means = get_experiment_rdfs(base_path)
    
    if len(exp_means) > 0:
        max_r = exp_means.shape[1]
        r = np.arange(max_r) * scale
        
        # 実験間の平均と標準誤差を計算
        rdf_mean = np.nanmean(exp_means, axis=0)
        rdf_sem = np.nanstd(exp_means, axis=0) / np.sqrt(len(exp_means))
        
        # ポテンシャルと、誤差伝播を用いたポテンシャルの標準誤差を計算
        # U(r) = -ln(g(r))
        # 誤差 dU = dg / g
        with np.errstate(divide='ignore', invalid='ignore'):
            potential_mean = -np.log(rdf_mean)
            potential_sem = rdf_sem / rdf_mean
            
        # グラフの描画
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots()
            show_plot = True
            
        ax.plot(r, potential_mean, label=f'{diameter} μm', color=color, marker=marker)
        ax.fill_between(r, potential_mean - potential_sem, potential_mean + potential_sem, color=color, alpha=0.3)
        
        # 基準線
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel('Distance $r$ [μm]')
        ax.set_ylabel('Effective Potential $U_\mathrm{eff}(r)$')
        #ax.set_title('Effective Potential')
        ax.legend()
        ax.set_xlim(0, 20)

        if show_plot:
            plt.show()
    else:
        print(f"有効な RDF.zarr が見つかりませんでした: {base_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot and save RDF and Effective Potential.')
    parser.add_argument('base_path', type=str, nargs='?', default=None, help='Base path glob pattern (e.g., "/Volumes/data/Sasaki/MTsingleBeads/beads1um/*/*")')
    parser.add_argument('--plot_all', action='store_true', help='Plot all predefined bead sizes (0.6um to 20um) in a single graph')
    parser.add_argument('--diameter', type=float, default=1.0, help='Diameter of the beads in um (default: 1.0)')
    parser.add_argument('--output_dir', type=str, default='/Volumes/data/Sasaki/MTsingleBeads/figure', help='Output directory for saving figures (default: figures/)')
    parser.add_argument('--color', type=str, default='C0', help='Color for the plot (default: C0)')
    parser.add_argument('--marker', type=str, default='^', help='Marker for the plot (default: ^)')
    parser.add_argument('--scale', type=float, default=0.11, help='Scale um/px (default: 0.11)')
    
    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # スタイルの適用
    style_path = os.path.join(os.path.dirname(__file__), 'my_style.mplstyle')
    if os.path.exists(style_path):
        plt.style.use(style_path)

    if args.plot_all:
        beads = [
            (0.6, "/Volumes/data/Sasaki/MTsingleBeads/beads06um/*/*", "C0", "^"),
            (1.0, "/Volumes/data/Sasaki/MTsingleBeads/beads1um/*/*", "C1", "o"),
            (3.0, "/Volumes/data/Sasaki/MTsingleBeads/beads3um/*/*", "C2", "d"),
            (5.0, "/Volumes/data/Sasaki/MTsingleBeads/beads5um/*/*", "C3", 10),
            (7.0, "/Volumes/data/Sasaki/MTsingleBeads/beads7um/*/*", "C4", 11),
            #(20.0, "/Volumes/data/Sasaki/MTsingleBeads/beads20um/*/*", "C5", "s"),
        ]
        
        # すべてのRDFを1つのグラフにプロット
        fig_rdf, ax_rdf = plt.subplots()
        for diameter, path, color, marker in beads:
            plot_RDF(path, ax=ax_rdf, diameter=diameter, color=color, marker=marker, scale=args.scale)
        rdf_out = os.path.join(args.output_dir, 'RDF_all_small.png')
        fig_rdf.savefig(rdf_out, bbox_inches='tight')
        plt.close(fig_rdf)
        print(f"Saved combined RDF plot to {rdf_out}")
        
        # すべてのポテンシャルを1つのグラフにプロット
        fig_pot, ax_pot = plt.subplots()
        for diameter, path, color, marker in beads:
            plot_potential(path, ax=ax_pot, diameter=diameter, color=color, marker=marker, scale=args.scale)
        pot_out = os.path.join(args.output_dir, 'Potential_all_small.png')
        fig_pot.savefig(pot_out, bbox_inches='tight')
        plt.close(fig_pot)
        print(f"Saved combined Potential plot to {pot_out}")
        
    elif args.base_path:
        # RDFの保存
        fig_rdf, ax_rdf = plt.subplots()
        plot_RDF(args.base_path, ax=ax_rdf, diameter=args.diameter, color=args.color, marker=args.marker, scale=args.scale)
        rdf_out = os.path.join(args.output_dir, f'RDF_{args.diameter}um.svg')
        fig_rdf.savefig(rdf_out, bbox_inches='tight')
        plt.close(fig_rdf)
        print(f"Saved RDF plot to {rdf_out}")
        
        # ポテンシャルの保存
        fig_pot, ax_pot = plt.subplots()
        plot_potential(args.base_path, ax=ax_pot, diameter=args.diameter, color=args.color, marker=args.marker, scale=args.scale)
        pot_out = os.path.join(args.output_dir, f'Potential_{args.diameter}um.svg')
        fig_pot.savefig(pot_out, bbox_inches='tight')
        plt.close(fig_pot)
        print(f"Saved Potential plot to {pot_out}")
    else:
        print("エラー: base_pathを指定するか、--plot_all オプションを使用してください。")

if __name__ == '__main__':
    main()
