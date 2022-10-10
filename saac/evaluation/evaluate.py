import pandas as pd
import numpy as np
from saac.statistics import ks2sample_test
from scipy.stats import ranksums,f_oneway,binomtest
from eval_utils import rgb_sorter, rgb_intensity

def evaluate_by_occupation(occupation_results=''):
	occ_res_all = pd.read_csv(occupation_results).sort_values('a_median')
	print(f'Total rows: {len(occ_res_all)}')
	print('Counts of sampled wage categories for median annual wage for all possible gender detected values ')

	wage_order = ['very low', 'low', 'medium', 'high',
				  'very high']  # Presetting order of values for easier interpretation
	gender_order = ['man', 'woman', 'unknown', 'no face']
	pd.crosstab(occ_res_all['gender_detected_val'], occ_res_all['wage_val']).reindex(gender_order)[wage_order]
	# For the case of this evaluation we will not be including images where a face could not be detected
	# or where the gender could not be determined

	occ_res = occ_res_all[~occ_res_all['gender_detected_val'].isin(['unknown', 'no face'])]
	print(f"Total rows after removing faceless and unknown gender detected results: {len(occ_res)}")
	o = [x for x in ks2sample_test(occ_res, group_col='gender_detected_val', value_col='a_median')]
	mask_male = occ_res['gender_detected_cat'] == 4
	mask_female = occ_res['gender_detected_cat'] == 3
	male_salary = occ_res[mask_male]['a_median'].median()
	female_salary = occ_res[mask_female]['a_median'].median()

	print(f"Median salary for male faces: {male_salary:0.2f}")
	print(f"Median salary for female faces: {female_salary:0.2f}")

	wcox_results = ranksums(occ_res[mask_male]['a_median'], occ_res[mask_female]['a_median'])
	print(wcox_results.statistic)
	print(wcox_results.pvalue)


def evaluate_by_adjectives(adjective_results=''):
	tda_res_all = pd.read_csv(adjective_results)
	print(f'Total rows: {len(tda_res_all)}')

	sentcheck = tda_res_all[tda_res_all['tda_compound'] == tda_res_all['prompt_compound']]
	print(f'Total rows where tda sentiment is equal to prompt sentiment : {len(sentcheck)}')

	print('Counts of sampled sentiment categories for all possible gender detected values ')
	sentiment_order = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
	gender_order = ['man', 'woman', 'unknown', 'no face']
	pd.crosstab(tda_res_all['gender_detected_val'], tda_res_all['tda_sentiment_val']).reindex(gender_order)[
		sentiment_order]
	tda_res = tda_res_all[~tda_res_all['gender_detected_val'].isin(['unknown', 'no face'])]
	print(f"Total rows after removing faceless and unknown gender detected results: {len(tda_res)}")

	t = [x for x in ks2sample_test(tda_res, group_col='gender_detected_val', value_col='tda_compound')]


def evaluate_skin_by_adjectives(tda_res):
	n_bins = 21
	tda_count, tda_division = np.histogram(tda_res['tda_compound'], bins=n_bins)
	tda_hist = tda_res.hist(column='tda_compound', bins=n_bins)
	for idx in range(1, len(tda_division)):
		if idx + 1 == len(tda_division):
			mask = (tda_res['tda_compound'] >= tda_division[idx - 1]) & (tda_res['tda_compound'] <= tda_division[idx])
		else:
			mask = (tda_res['tda_compound'] >= tda_division[idx - 1]) & (tda_res['tda_compound'] < tda_division[idx])

		sorted_rgb = rgb_sorter(tda_res[mask]['skincolor'].apply(eval))
		fig, ax = plt.subplots(1, 1)

		tda_count, tda_division = np.histogram(tda_res['tda_compound'], bins=n_bins)

		all_rgb_intensities = []

		for idx in range(1, len(tda_division)):
			if idx + 1 == len(tda_division):
				mask = (tda_res['tda_compound'] >= tda_division[idx - 1]) & (
							tda_res['tda_compound'] <= tda_division[idx])
			else:
				mask = (tda_res['tda_compound'] >= tda_division[idx - 1]) & (
							tda_res['tda_compound'] < tda_division[idx])

			if sum(mask) <= 0:
				continue

			rgb_intensities = tda_res[mask]['skincolor'].apply(eval).apply(rgb_intensity)
			all_rgb_intensities.append(list(rgb_intensities.values))


		F, p = f_oneway(*all_rgb_intensities)
		print(F)
		print(p)

def evaluate_gender_by_skincolor(tda_res):


	gender_sig = binomtest(len(tda_res[tda_res['gender_detected_val'] == 'woman']), n=len(tda_res), p=0.5)
	print(f"p-value of hypothesis that both men and women are represented equally: {gender_sig}")

def evaluate(filepath):
	evaluate_by_adjectives()
	evaluate_by_occupation()

if __name__=='__main__':
	evaluate()