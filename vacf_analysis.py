#!/usr/bin/env python3
"""
vacf_analysis.py

貨物微粒子（蛍光ビーズ）の自己相関関数を一括解析・可視化するスクリプトです。
ビーズ 0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20.0μm の全条件を対象に、
- 速度ベクトルの自己相関 (VACF: Velocity Autocorrelation Function)
- 配向方向の自己相関 (OACF: Orientation Autocorrelation Function)
- 速さスカラーの自己相関 (SACF: Speed Autocorrelation Function)
を一括で計算し、SVG/PNGプロットおよびCSVサマリーを出力します。
"""

from libs.vacf_analysis import main

if __name__ == "__main__":
    main()
