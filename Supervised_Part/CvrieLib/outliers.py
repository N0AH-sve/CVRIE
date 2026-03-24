##
## EPITECH PROJECT, 2026
## G-AIA-400-NAN-4-1-cvrie-2
## File description:
## outliers
##

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_all_named(directory, target_size, pipeline_builder, extension=".jpg"):
	"""		Load all images from a directory and return flattened arrays with names.
		Work like load_... in the notebook

	Args:
		directory (str): Directory containing image files.
		target_size (tuple): Target shape used by the preprocessing pipeline.
		pipeline_builder (callable): Function receiving target_size and returning
			a fitted-transform compatible pipeline.
		extension (str): File extension filter (default: .jpg).

	Returns:
		tuple[np.ndarray, list[str]]: (flattened_images, image_names)
	"""
	files = sorted([f for f in os.listdir(directory) if f.lower().endswith(extension)])
	pipeline = pipeline_builder(target_size=target_size)
	data_list, names_list = [], []

	for file_name in files:
		try:
			image = pipeline.fit_transform(os.path.join(directory, file_name)).flatten()
			data_list.append(image)
			names_list.append(file_name)
		except Exception:
			continue
	return np.array(data_list, dtype=np.float32), names_list


def get_model_accuracy(x_frac, rows_frac, x_nfrac, rows_nfrac, test_size=0.2, random_state=5, n_estimators=20, max_depth=5):
	"""     Train a RandomForest on the given subsets and return test accuracy.

	Args:
		x_frac (np.ndarray): Flattened fractured images.
		rows_frac (list[int]): Row indices to use from x_frac.
		x_nfrac (np.ndarray): Flattened not fractured images.
		rows_nfrac (list[int]): Row indices to use from x_nfrac.
		test_size (float): Validation split ratio.
		random_state (int): Seed for reproducibility.
		n_estimators (int): RandomForest n_estimators.
		max_depth (int | None): RandomForest max_depth.

	Returns:
		float: Accuracy of the model on the test set.
	"""
	x_sub = np.vstack([x_frac[rows_frac], x_nfrac[rows_nfrac]])
	y_sub = np.array([1] * len(rows_frac) + [0] * len(rows_nfrac), dtype=np.int8)

	x_train, x_test, y_train, y_test = train_test_split(x_sub, y_sub, test_size=test_size, random_state=random_state, stratify=y_sub)
	model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state)
	model.fit(x_train, y_train)
	return model.score(x_test, y_test)


def compute_thresholds(df, q_frac, q_nfrac, idx_frac, idx_nfrac):
	"""     Compute thresholds and valid row indices for given quantiles.
	Args:
		df (pd.DataFrame): Dataframe containing columns Class, Mean, Name.
		q_frac (float): Quantile for fractured class.
		q_nfrac (float): Quantile for not fractured class.
	Returns:
		tuple[float, float, list[int], list[int]]:
			(thresh_frac, thresh_nfrac, rows_frac, rows_nfrac)
	"""
	thresh_frac = df[df["Class"] == "fractured"]["Mean"].quantile(q_frac)
	thresh_nfrac = df[df["Class"] == "not_fractured"]["Mean"].quantile(q_nfrac)

	valid_frac = set(df[(df["Class"] == "fractured") & (df["Mean"] <= thresh_frac)]["Name"])
	valid_nfrac = set(df[(df["Class"] == "not_fractured") & (df["Mean"] <= thresh_nfrac)]["Name"])
	rows_frac = [idx_frac[name] for name in valid_frac if name in idx_frac]
	rows_nfrac = [idx_nfrac[name] for name in valid_nfrac if name in idx_nfrac]

	return thresh_frac, thresh_nfrac, rows_frac, rows_nfrac


def init_idx_pairs(names_frac, names_nfrac, q_frac_vals, q_nfrac_vals, df):
	"""	 Initialize index mappings and quantile pairs for optimization.
	Args:
		names_frac (list[str]): Names matching rows in x_frac.
		names_nfrac (list[str]): Names matching rows in x_nfrac.
		q_frac_vals (iterable): Quantiles tested for fractured class.
		q_nfrac_vals (iterable): Quantiles tested for not fractured class.
		df (pd.DataFrame): Dataframe containing columns Class, Mean, Name.
	Returns:
		tuple[dict, dict, int, int, list[tuple[float, float]], int]: 
		(idx_frac, idx_nfrac, total_frac, total_nfrac, pairs, total_pairs)
	"""
	idx_frac = {name: i for i, name in enumerate(names_frac)}
	idx_nfrac = {name: i for i, name in enumerate(names_nfrac)}

	total_frac = df[df["Class"] == "fractured"]["Name"].nunique()
	total_nfrac = df[df["Class"] == "not_fractured"]["Name"].nunique()

	pairs = [(float(qf), float(qnf)) for qf in q_frac_vals for qnf in q_nfrac_vals]
	total_pairs = len(pairs)
	print(f"Starting quantile optimization with {total_pairs} combinations...")
	return idx_frac, idx_nfrac, total_frac, total_nfrac, pairs, total_pairs


def optimize_outlier_quantiles(	df,	x_frac,	names_frac,	x_nfrac, names_nfrac, q_frac_vals, q_nfrac_vals, test_size=0.2,
	random_state=5,	min_per_class=10, n_estimators=20, max_depth=5, progress_every=20,):
	"""		Run a quantile grid search and evaluate model accuracy on each pair.

	Args:
		df (pd.DataFrame): Dataframe containing columns Class, Mean, Name.
		x_frac (np.ndarray): Flattened fractured images.
		names_frac (list[str]): Names matching rows in x_frac.
		x_nfrac (np.ndarray): Flattened not fractured images.
		names_nfrac (list[str]): Names matching rows in x_nfrac.
		q_frac_vals (iterable): Quantiles tested for fractured class.
		q_nfrac_vals (iterable): Quantiles tested for not fractured class.
		test_size (float): Validation split ratio.
		random_state (int): Seed for reproducibility.
		min_per_class (int): Skip pairs with less than this number of samples.
		n_estimators (int): RandomForest n_estimators.
		max_depth (int | None): RandomForest max_depth.
		progress_every (int): Print progress every N combinations.

	Returns:
		tuple[pd.DataFrame, pd.Series, list[tuple[float, float]]]:
			(results_dataframe, best_row, all_pairs)
	"""
	idx_frac, idx_nfrac, total_frac, total_nfrac, pairs, total_pairs = init_idx_pairs(names_frac, names_nfrac, q_frac_vals, q_nfrac_vals, df)
	
	results = []
	for idx, (q_frac, q_nfrac) in enumerate(pairs, start=1):
		thresh_frac, thresh_nfrac, rows_frac, rows_nfrac = compute_thresholds(df, q_frac, q_nfrac, idx_frac, idx_nfrac)

		if len(rows_frac) < min_per_class or len(rows_nfrac) < min_per_class:
			continue
		
		score = get_model_accuracy(x_frac, rows_frac, x_nfrac, rows_nfrac, test_size=test_size, random_state=random_state, n_estimators=n_estimators, max_depth=max_depth)

		results.append(
			{
				"q_frac": round(q_frac, 2),
				"q_nfrac": round(q_nfrac, 2),
				"accuracy": score,
				"removed_frac": total_frac - len(rows_frac),
				"removed_nfrac": total_nfrac - len(rows_nfrac),
				"total_removed": (total_frac - len(rows_frac))
				+ (total_nfrac - len(rows_nfrac)),
			}
		)

		if progress_every > 0 and (idx % progress_every == 0 or idx == total_pairs):
			print(f"  Progression: {idx}/{total_pairs}")
	results_df = pd.DataFrame(results)
	if results_df.empty:
		raise ValueError("No valid quantile pairs found with the given min_per_class threshold.")
	best_row = results_df.loc[results_df["accuracy"].idxmax()]
	return results_df, best_row, pairs


def run_outlier_quantile_optimization( df, pipeline_builder, fractured_dir="Dataset/fractured/", not_fractured_dir="Dataset/not_fractured/",
	target_size=(64, 64), q_frac_vals=None,	q_nfrac_vals=None, test_size=0.2, random_state=5, min_per_class=10,	n_estimators=20,
	max_depth=5, progress_every=20):
	"""		Wrapper that loads datasets and runs quantile optimization.

	Returns:
		tuple[pd.DataFrame, pd.Series, dict]:
			(results_dataframe, best_row, cache)
			cache contains loaded arrays and names to avoid reloading later.
	"""
	print(f"Loading datasets with target size {target_size}...")
	x_frac, names_frac = load_all_named(fractured_dir, target_size, pipeline_builder)
	x_nfrac, names_nfrac = load_all_named(not_fractured_dir, target_size, pipeline_builder)

	if q_frac_vals is None:
		q_frac_vals = np.round(np.arange(0.90, 1.001, 0.01), 2)
	if q_nfrac_vals is None:
		q_nfrac_vals = np.round(np.arange(0.87, 1.001, 0.01), 2)

	results_df, best_row, pairs = optimize_outlier_quantiles(
		df=df, x_frac=x_frac, names_frac=names_frac, x_nfrac=x_nfrac, names_nfrac=names_nfrac,
		q_frac_vals=q_frac_vals, q_nfrac_vals=q_nfrac_vals, test_size=test_size,
		random_state=random_state, min_per_class=min_per_class, n_estimators=n_estimators,
		max_depth=max_depth, progress_every=progress_every
	)

	cache = { "x_frac": x_frac, "names_frac": names_frac, "x_nfrac": x_nfrac, "names_nfrac": names_nfrac, "pairs": pairs, "target_size": target_size }
	return results_df, best_row, cache
