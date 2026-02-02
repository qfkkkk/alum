import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_SELECTED_COLUMNS = [
	"1#流量计投加药耗",
	"1#出水浊度",
	"1#进水流量",
	"1#反应池浊度",
	"前池水下温度",
	"前池水面温度",
	"2#反应池pH",
]

def load_data(path, datetime_col=None):
	df = pd.read_csv(path)

	# 如果存在分开的日期/小时/分钟列，则优先合并为完整的时间列
	date_parts = {"DateDay", "DateHour", "Minute"}
	if date_parts.issubset(set(df.columns)) and (datetime_col is None or datetime_col == "DateDay"):
		try:
			day = df['DateDay'].astype(str).str.strip()
			hour = df['DateHour'].astype(str).str.zfill(2).str.strip()
			minute = df['Minute'].astype(str).str.zfill(2).str.strip()
			dt_str = day + ' ' + hour + ':' + minute
			# 尝试解析合并后的字符串为 datetime
			df['DateTime'] = pd.to_datetime(dt_str, errors='coerce')
			# 仅在解析成功（非全为 NaT）时使用合并列
			if df['DateTime'].notna().any():
				datetime_col = 'DateTime'
		except Exception:
			# 若合并解析失败，则继续后续的自动检测逻辑
			pass
	if datetime_col is None:
		raise ValueError("未能找到时间列，请在参数中指定 datetime_col")

	# 转换为 datetime 并设置为索引
	df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
	df = df.set_index(datetime_col).sort_index()
	return df


def initial_analysis(df, cols, out_dir):
	os.makedirs(out_dir, exist_ok=True)
	report = []
	available = [c for c in cols if c in df.columns]
	missing = [c for c in cols if c not in df.columns]
	print(f"可分析列: {available}")
	if missing:
		print(f"警告：以下列未找到并将被跳过: {missing}")
	for c in available:
		s = df[c]
		stats = s.describe()
		nulls = s.isna().sum()
		skew = s.skew()
		kurt = s.kurt()
		report.append(
			{
				"column": c,
				"count": int(stats.get("count", 0)),
				"mean": float(stats.get("mean", np.nan)),
				"std": float(stats.get("std", np.nan)),
				"min": float(stats.get("min", np.nan)),
				"25%": float(stats.get("25%", np.nan)),
				"50%": float(stats.get("50%", np.nan)),
				"75%": float(stats.get("75%", np.nan)),
				"max": float(stats.get("max", np.nan)),
				"nulls": int(nulls),
				"skew": float(skew),
				"kurt": float(kurt),
			}
		)
	rpt_df = pd.DataFrame(report)
	rpt_df.to_csv(os.path.join(out_dir, "initial_analysis_report.csv"), index=False)
	print(f"初步统计报告已保存: {os.path.join(out_dir, 'initial_analysis_report.csv')}")
	return available


def detect_and_replace_anomalies(df, cols, window_minutes=120, z_thresh=3.0, out_dir="./figures"):
	os.makedirs(out_dir, exist_ok=True)
	anomalies = []
	df_clean = df.copy()
	# 需要保证索引为DatetimeIndex
	if not isinstance(df_clean.index, pd.DatetimeIndex):
		raise ValueError("DataFrame 必须以时间索引 (DatetimeIndex)")

	for c in cols:
		if c not in df_clean.columns:
			continue
		series = df_clean[c]
		# time-based rolling window
		rolling_mean = series.rolling(f"{window_minutes}min", min_periods=1).mean()
		rolling_std = series.rolling(f"{window_minutes}min", min_periods=1).std()
		z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
		mask = z.abs() > z_thresh
		detected = series[mask].dropna()
		# helper: safely extract a scalar value for a given timestamp
		def _safe_get(series_like, ts):
			try:
				v = series_like.loc[ts]
			except Exception:
				return np.nan
			# 如果返回的是 Series/array（可能由重复索引或切片引起），取第一个元素
			if isinstance(v, (pd.Series, pd.DataFrame)):
				if v.size == 0:
					return np.nan
				v = v.iloc[0]
			elif isinstance(v, (list, tuple, np.ndarray)):
				if len(v) == 0:
					return np.nan
				v = v[0]
			try:
				return float(v) if pd.notna(v) else np.nan
			except Exception:
				return np.nan

		for ts, val in detected.items():
			anomalies.append(
				{
					"column": c,
					"timestamp": ts,
					"original": float(val) if pd.notna(val) else np.nan,
					"rolling_mean": _safe_get(rolling_mean, ts),
					"rolling_std": _safe_get(rolling_std, ts),
					"z_score": _safe_get(z, ts),
				}
			)
		# 用窗口均值替换异常值（当均值可用时）
		replace_mask = mask & rolling_mean.notna()
		df_clean.loc[replace_mask, c] = rolling_mean.loc[replace_mask]
		print(f"{c}: 识别到 {int(mask.sum())} 个异常, 替换 {int(replace_mask.sum())} 个")

	anom_df = pd.DataFrame(anomalies)
	anom_path = os.path.join(out_dir, "anomaly_report.csv")
	anom_df.to_csv(anom_path, index=False)
	print(f"异常值报告已保存: {anom_path}")
	return df_clean, anom_df


def aggregate_and_plot(df, cols, out_dir="figure_result", freq='60T'):
	os.makedirs(out_dir, exist_ok=True)
	# 按照 freq 聚合
	agg = df[cols].resample(freq).mean()
	agg_counts = df[cols].resample(freq).count()
	agg.to_csv(os.path.join(out_dir, "aggregated_mean.csv"))
	agg_counts.to_csv(os.path.join(out_dir, "aggregated_count.csv"))
	print(f"聚合数据已保存: {os.path.join(out_dir, 'aggregated_mean.csv')}")

	# 为每个列画折线图与直方图
	for c in cols:
		if c not in agg.columns:
			continue
		plt.figure(figsize=(12, 5))
		ax = plt.gca()
		sns.lineplot(data=agg[c], ax=ax)
		stats = agg[c].describe()
		text = (
			f"n={int(stats['count'])}\nmean={stats['mean']:.3f}\nstd={stats['std']:.3f}\nmin={stats['min']:.3f}\nmax={stats['max']:.3f}"
		)
		ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
		ax.set_title(f"{c} - {freq} 聚合折线图")
		ax.set_xlabel('时间')
		ax.set_ylabel(c)
		plt.tight_layout()
		out_png = os.path.join(out_dir, f"timeseries_{c}.png")
		plt.savefig(out_png)
		plt.close()

		# 直方图（频率分布）
		plt.figure(figsize=(7, 4))
		sns.histplot(agg[c].dropna(), kde=True)
		plt.title(f"{c} - {freq} 聚合频率分布")
		plt.xlabel(c)
		stats_text = (
			f"n={int(stats['count'])} mean={stats['mean']:.3f} std={stats['std']:.3f}")
		plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
		out_hist = os.path.join(out_dir, f"hist_{c}.png")
		plt.tight_layout()
		plt.savefig(out_hist)
		plt.close()
		print(f"已保存图像: {out_png}, {out_hist}")

	return agg


def other_modeling_stats(df, cols, out_dir="figure_result"):
	os.makedirs(out_dir, exist_ok=True)
	sub = df[cols].copy()
	corr = sub.corr()
	corr.to_csv(os.path.join(out_dir, "correlation_matrix.csv"))
	print(f"相关性矩阵已保存: {os.path.join(out_dir, 'correlation_matrix.csv')}")
	# 绘制热力图
	plt.figure(figsize=(8, 6))
	sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag')
	plt.title('Selected Columns Correlation')
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
	plt.close()

	# 缺失值分布
	missing = sub.isna().sum().rename('missing_count')
	missing.to_csv(os.path.join(out_dir, "missing_counts.csv"))
	print(f"缺失值统计已保存: {os.path.join(out_dir, 'missing_counts.csv')}")


def main():
	parser = argparse.ArgumentParser(description="处理CSV数据：统计、异常检测、聚合与绘图")
	parser.add_argument("input", help="输入CSV文件路径", nargs='?', default="pro_data/merged_data.csv")
	parser.add_argument("--cols", help="用逗号分隔的列名（覆盖默认）", default=None)
	parser.add_argument("--out", help="输出目录", default="figure_result")
	parser.add_argument("--window", help="异常检测滚动窗口(分钟)", type=int, default=120)
	parser.add_argument("--freq", help="聚合频率(例如60T)", default="60T")
	parser.add_argument("--z", help="Z阈值 (默认为3.0)", type=float, default=3.0)
	args = parser.parse_args()

	df = load_data(args.input)
	if args.cols:
		cols = args.cols.split(',')
	else:
		cols = DEFAULT_SELECTED_COLUMNS

	available = initial_analysis(df, cols, args.out)
	cleaned, anom = detect_and_replace_anomalies(df, available, window_minutes=args.window, z_thresh=args.z, out_dir=args.out)
	agg = aggregate_and_plot(cleaned, available, out_dir=args.out, freq=args.freq)
	other_modeling_stats(cleaned, available, out_dir=args.out)
	print("处理完成。")


if __name__ == '__main__':
	main()
