import warnings
import copy
import patsy
import numpy as np
import pandas as pd
from scipy.stats import logistic
import statsmodels.api as sm
import statsmodels.formula.api as smf

class DoubleCrossfitTMLE:
    """Implementation of the targeted maximum likelihood estimator with a double cross-fit procedure happening in the
    background

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    treatment : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame
    random_state : int, optional
        Attempt to add a seed for reproducibility of all the algorithms. Not confirmed as stable yet

    Notes
    -----
    To be added to the zEpid library in the near future. TODO's throughout correspond to items needed to be added to
    the zEpid implementation

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> dctmle = DoubleCrossfitTMLE(data, 'X', 'Y')
    >>> dctmle.treatment_model('Z', LogisticRegression(penalty='none', solver='lbfgs'), bound=0.01)
    >>> dctmle.outcome_model('X + Z', LogisticRegression(penalty='none', solver='lbfgs'))
    >>> dctmle.fit(resamples=100, method='median')
    """
    def __init__(self, df, treatment, outcome, random_state=None):
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, CrossfitTMLE will drop all missing data. "
                          "Crossfit TMLE will fit " + str(df.dropna().shape[0]) + ' of ' +
                          str(df.shape[0]) + ' observations', UserWarning)
        self.df = df.copy().dropna().reset_index()
        self.treatment = treatment
        self.outcome = outcome
        if random_state is None:
            self._seed_ = np.random.randint(1, 10000)
        else:
            self._seed_ = random_state

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False
        self._gbounds = None

        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

    def treatment_model(self, covariates, estimator, bound=False):
        """Specify the treatment nuisance model variables and estimator(s) to use. These parameters are held until usage
        in the .fit() function. These approaches are for each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        bound : float, list, optional
            Whether to bound predicted probabilities. Default is False, which does not bound
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True
        self._gbounds = bound

    def outcome_model(self, covariates, estimator):
        """Specify the outcome nuisance model variables and estimator(s) to use. These parameters are held until usage
        in the .fit() function. These approaches are for each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self, resamples, method='median'):
        """Runs the double-crossfit estimation procedure with targeted maximum likelihood estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the average causal effect from each of the different splits. Median is
        used as the default since it is more stable.

        Confidence intervals come from influences curves and incorporates the within-split variance and between-split
        variance.

        Parameters
        ----------
        resamples : int
            Number of times to repeat the sample-splitting estimation process. It is recommended to use at least 100.
            Note that this algorithm can take a long time to run for high values of this parameter. Be sure to test out
            run-times on small numbers first
        method : str, optional
            Method to obtain point estimates and standard errors. Median method takes the median (which is more robust)
            and the mean takes the mean. It has been remarked that the median is preferred, since it is more stable to
            extreme outliers, which may happen in finite samples
        """
        # Creating blank lists
        rd_point = []
        rd_var = []

        # Conducts the re-sampling procedure
        for j in range(resamples):
            split_samples = _sample_split_(self.df, seed=None)

            # Estimating (lots of functions happening in the background
            result = self._single_crossfit_(sample_split=split_samples)

            # Appending this particular split's results
            rd_point.append(result[0])
            rd_var.append(result[1])

        if method == 'median':
            self.risk_difference = np.median(rd_point)
            self.risk_difference_se = np.sqrt(np.median(rd_var + (rd_point - self.risk_difference)**2))
        elif method == 'mean':
            self.risk_difference = np.mean(rd_point)
            self.risk_difference_se = np.sqrt(np.mean(rd_var + (rd_point - self.risk_difference)**2))
        else:
            raise ValueError("Either 'mean' or 'median' must be selected for the pooling of repeated sample splits")

        self.risk_difference_ci = (self.risk_difference - 1.96*self.risk_difference_se,
                                   self.risk_difference + 1.96*self.risk_difference_se)

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('The treatment and outcome models must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('                       Double-Crossfit TMLE')
        print('======================================================================')
        print('----------------------------------------------------------------------')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('SE(RD):          ', round(float(self.risk_difference_se), decimal))
        print('95% CL:          ', round(float(self.risk_difference_ci[0]), decimal),
              round(float(self.risk_difference_ci[1]), decimal))
        print('======================================================================')

    def _single_crossfit_(self, sample_split):
        """Background function that runs a single crossfit of the split samples
        """

        # Estimating treatment function
        a_models = _treatment_nuisance_(treatment=self.treatment, estimator=self._a_estimator,
                                        samples=sample_split, covariates=self._a_covariates)

        # Estimating outcome function
        y_models = _outcome_nuisance_(outcome=self.outcome, estimator=self._y_estimator,
                                      samples=sample_split, covariates=self._y_covariates)

        # Calculating predictions for each sample split and each combination
        s1_predictions = self._generate_predictions(sample_split[0], a_model_v=a_models[1], y_model_v=y_models[2])
        s2_predictions = self._generate_predictions(sample_split[1], a_model_v=a_models[2], y_model_v=y_models[0])
        s3_predictions = self._generate_predictions(sample_split[2], a_model_v=a_models[0], y_model_v=y_models[1])

        # Observed values of treatment and outcome
        y_obs = np.append(np.asarray(sample_split[0][self.outcome]),
                          np.append(np.asarray(sample_split[1][self.outcome]),
                                    np.asarray(sample_split[2][self.outcome])))
        a_obs = np.append(np.asarray(sample_split[0][self.treatment]),
                          np.append(np.asarray(sample_split[1][self.treatment]),
                                    np.asarray(sample_split[2][self.treatment])))

        split_index = np.asarray([0]*sample_split[0].shape[0] + [1]*sample_split[1].shape[0] +
                                 [2]*sample_split[2].shape[0])

        # Stacking predicted Pr(A=)
        pred_a_array = np.append(s1_predictions[0], np.append(s2_predictions[0], s3_predictions[0]))
        if self._gbounds:  # Bounding g-model if requested
            pred_a_array = _bounding_(pred_a_array, bounds=self._gbounds)

        # Stacking predicted outcomes under each treatment plan
        pred_treat_array = np.append(s1_predictions[1], np.append(s2_predictions[1], s3_predictions[1]))
        pred_none_array = np.append(s1_predictions[2], np.append(s2_predictions[2], s3_predictions[2]))

        ate, var_ate = self._tmle_calculator(y_obs=y_obs, a=a_obs,
                                             qaw=np.where(a_obs == 1, pred_treat_array, pred_none_array),
                                             qa1w=pred_treat_array, qa0w=pred_none_array,
                                             g1w=pred_a_array, g0w=1 - pred_a_array,
                                             splits=split_index, continuous=False)
        return ate, var_ate

    def _generate_predictions(self, sample, a_model_v, y_model_v):
        """Generates predictions from fitted functions (in background of _single_crossfit()
        """
        s = sample.copy()

        # Predicting Pr(A=1|L)
        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', s))
        a_pred = _ml_predictor(xdata, fitted_algorithm=a_model_v)

        # Predicting E(Y|A=1, L)
        s[self.treatment] = 1
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_treat = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        # Predicting E(Y|A=0, L)
        s[self.treatment] = 0
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_none = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        return a_pred, y_treat, y_none

    @staticmethod
    def _tmle_calculator(y_obs, a, qaw, qa1w, qa0w, g1w, g0w, splits, continuous=False):
        """Background targeting step from g-model and Q-model
        """
        h1w = a / g1w
        h0w = -(1 - a) / g0w
        haw = h1w + h0w

        qstar = []
        qstar1 = []
        qstar0 = []

        # Calculating overall estimate
        for i in [0, 1, 2]:
            yb_ = y_obs[splits == i]
            g1s = g1w[splits == i]
            g0s = g0w[splits == i]
            q1s = qa1w[splits == i]
            q0s = qa0w[splits == i]
            qas = qaw[splits == i]
            h1s = h1w[splits == i]
            h0s = h0w[splits == i]

            # Targeting model
            f = sm.families.family.Binomial()
            log = sm.GLM(yb_, np.column_stack((h1s, h0s)), offset=np.log(probability_to_odds(qas)),
                         family=f, missing='drop').fit()
            epsilon = log.params

            qstar1 = np.append(qstar1, logistic.cdf(np.log(probability_to_odds(q1s)) + epsilon[0] / g1s))
            qstar0 = np.append(qstar0, logistic.cdf(np.log(probability_to_odds(q0s)) - epsilon[1] / g0s))
            qstar = np.append(qstar, log.predict(np.column_stack((h1s, h0s)), offset=np.log(probability_to_odds(qas))))

        # TODO bounding bit if continuous
        if continuous:
            raise ValueError("Not completed yet")
            # TODO I do an unbounding step here for the outcomes if necessary
            # y_obs = self._unit_unbound(y_bound, mini=self._continuous_min, maxi=self._continuous_max)
            # qstar = self._unit_unbound(qstar, mini=self._continuous_min, maxi=self._continuous_max)
            # qstar1 = self._unit_unbound(qstar1, mini=self._continuous_min, maxi=self._continuous_max)
            # qstar0 = self._unit_unbound(qstar0, mini=self._continuous_min, maxi=self._continuous_max)

        qstar_est = np.mean(qstar1 - qstar0)

        # Variance estimation
        var_rd = []
        for i in [0, 1, 2]:
            yu_ = y_obs[splits == i]
            qs1s = qstar1[splits == i]
            qs0s = qstar0[splits == i]
            qs = qstar[splits == i]
            has = haw[splits == i]

            ic = has * (yu_ - qs) + (qs1s - qs0s) - qstar_est
            var_rd.append(np.var(ic, ddof=1))

        return qstar_est, (np.mean(var_rd) / splits.shape[0])

###############################################################
# Background utility functions shared by estimators
def probability_to_odds(prob):
    """Converts given probability (proportion) to odds

    Parameters
    ---------------
    prob : float, NumPy array
        Probability or array of probabilities to transform into odds
    """
    return prob / (1 - prob)


def _bounding_(v, bounds):
    """Background function to perform bounding feature for inverse probability weights. Supports both symmetric
    and asymmetric bounding
    """
    if type(bounds) is float:  # Symmetric bounding
        if bounds < 0 or bounds > 1:
            raise ValueError('Bound value must be between (0, 1)')
        v = np.where(v < bounds, bounds, v)
        v = np.where(v > 1 - bounds, 1 - bounds, v)

    elif type(bounds) is str:  # Catching string inputs
        raise ValueError('Bounds must either be a float between (0, 1), or a collection of floats between (0, 1)')
    elif type(bounds) is int:  # Catching string inputs
        raise ValueError('Bounds must either be a float between (0, 1), or a collection of floats between (0, 1)')

    else:  # Asymmetric bounds
        if bounds[0] > bounds[1]:
            raise ValueError('Bound thresholds must be listed in ascending order')
        if len(bounds) > 2:
            warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                          'specified bounds are used by the bound statement. So only ' +
                          str(bounds[0:2]) + ' will be used', UserWarning)
        if type(bounds[0]) is str or type(bounds[1]) is str:
            raise ValueError('Bounds must be floats between (0, 1)')
        if (bounds[0] < 0 or bounds[1] > 1) or (bounds[0] < 0 or bounds[1] > 1):
            raise ValueError('Both bound values must be between (0, 1)')
        v = np.where(v < bounds[0], bounds[0], v)
        v = np.where(v > bounds[1], bounds[1], v)
    return v


def _sample_split_(data, seed):
    """Background function to split data into three non-overlapping pieces
    """
    n = int(data.shape[0] / 3)
    s1 = data.sample(n=n, random_state=seed)
    s2 = data.loc[data.index.difference(s1.index)].sample(n=n, random_state=seed)
    s3 = data.loc[data.index.difference(s1.index) & data.index.difference(s2.index)]
    return s1, s2, s3


def _ml_predictor(xdata, fitted_algorithm):
    """Background function to generate predictions of treatments
    """
    if hasattr(fitted_algorithm, 'predict_proba'):
        return fitted_algorithm.predict_proba(xdata)[:, 1]
    elif hasattr(fitted_algorithm, 'predict'):
        return fitted_algorithm.predict(xdata)


def _treatment_nuisance_(treatment, estimator, samples, covariates):
    """Procedure to fit the treatment ML
    """
    treatment_fit_splits = []
    for s in samples:
        # Using patsy to pull out the covariates
        xdata = np.asarray(patsy.dmatrix(covariates + ' - 1', s))
        ydata = np.asarray(s[treatment])

        # Fitting machine learner / super learner to each split
        est = copy.deepcopy(estimator)
        try:
            fm = est.fit(X=xdata, y=ydata)
            # print("Treatment model")
            # print(fm.summarize())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Adding model to the list of models
        treatment_fit_splits.append(fm)

    return treatment_fit_splits


def _outcome_nuisance_(outcome, estimator, samples, covariates):
    """Background function to generate predictions of outcomes
    """
    outcome_fit_splits = []
    for s in samples:
        # Using patsy to pull out the covariates
        xdata = np.asarray(patsy.dmatrix(covariates + ' - 1', s))
        ydata = np.asarray(s[outcome])

        # Fitting machine learner / super learner to each
        est = copy.deepcopy(estimator)
        try:
            fm = est.fit(X=xdata, y=ydata)
            # print("Outcome model")
            # print(fm.summarize())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Adding model to the list of models
        outcome_fit_splits.append(fm)

    return outcome_fit_splits
