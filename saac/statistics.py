import itertools
from scipy.stats import binom, fisher_exact, chi2_contingency, ks_2samp


def ks2sample_test(df, id_key=None, value_key=None, test=ks_2samp):
    """
    Applies the two sample Kolmogorov-Smirnov which tests if the empirical cumulative distribution of the samples
    are statistically similar or different. Takes dataframe and 2 dataframe columns test applies to n pair of groups in
    id_key column to compare their underlying distributions in the value_key column.
    Uses the 'twosided' value for H0 and Ha which sets the following:
    H0- the 2 distributions are identical, F(x)=G(x) for all x
    Ha- the 2 distributions are not identical
    With a confidence level of 95%, the H0 can be rejected in favor of the Ha if the p-value is less than 0.05
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#id1

    """
    # Group dataframe by identifier:
    g = df.groupby(id_key)
    # Generate all 2-combination of identifier:
    for k1, k2 in itertools.combinations(g.groups.keys(), 2):
        # Apply Statistical Test to grouped data:
        t = test(df.loc[g.groups[k1], value_key], df.loc[g.groups[k2], value_key], alternative='twosided')
        # Store identifier pair(s):
        res = {"id1": k1, "id2": k2}
        # Store statistics and p-value:
        res.update({k: getattr(t, k) for k in t._fields})
        # Yield result:
        yield res

def binomial_significance(n_true_1: int, n_true_2: int, n_false_1: int, n_false_2: int):
    table = [[n_true_1, n_true_2], [n_false_1, n_false_2]]

    if n_true_1/(n_true_1 + n_false_1) < n_true_2/(n_true_2 + n_false_2):
        alternative = "less"
    elif n_true_1/(n_true_1 + n_false_1) > n_true_2/(n_true_2 + n_false_2):
        alternative = "greater"
    else:
        alternative = "two-sided"

    # https://www.omnicalculator.com/statistics/fishers-exact-test
    # When to use Fisher's exact test?
    #   Choose the Fisher exact test rather than the chi-squared test if the sample size is small, the marginals
    #   are very uneven, or there is a small value (less than five) in one of the cells. This is because the
    #   approximation on which the chi-squared test is based might not be very accurate in such conditions.
    # When to use chi-square test?
    #   Since we have to compute factorials, Fisher's exact test is hard to calculate when the sample is large or
    #   the contingency table is well-balanced. Fortunately, there are the conditions under which the approximation
    #   used in the chi-squared test flourishes!
    # TODO: double check sample size limits/requirements
    if n_true_1 < 5 \
            or n_false_1 < 5 \
            or n_true_2 < 5 \
            or n_false_2 < 5 \
            or n_true_1 + n_false_1 < 50 \
            or n_true_2 + n_false_2 < 50:
        odds_ratio, p_value = fisher_exact(table, alternative=alternative)

        return {
            'Test': "fisher_exact",
            'odds_ratio': odds_ratio,
            'p-value': p_value,
            }

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    # An often quoted guideline for the validity of this calculation is that the test should be used only if the
    # observed and expected frequencies in each cell are at least 5.
    elif n_true_1 >= 5 \
            and n_false_1 >= 5 \
            and n_true_2 >= 5 \
            and n_false_2 >= 5:
        g, p, dof, expected = chi2_contingency(observed=table)

        return {
            'Test': "chi2_contingency",
            'g': g,
            'p-value': p,
            'dof': dof,
            'expected': expected,
            }

    else:
        print("WARNING: Underspecified statistical test")
        return None
