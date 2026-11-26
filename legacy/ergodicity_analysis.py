#!/usr/bin/env python3
"""
ergodicity_analysis.py

貨物微粒子（蛍光ビーズ）のエルゴード性破壊パラメータ (Ergodicity Breaking Parameter: EB)
および時間平均二乗変位 (Time-Averaged MSD: TAMSD) を
全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.00μm, 7.24μm, 20.0μm）で一括解析・可視化するスクリプトです。
"""

from libs.ergodicity_analysis import main

if __name__ == "__main__":
    main()
