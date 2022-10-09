from scipy.stats import binom, fisher_exact, chi2_contingency


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
