import numpy as np
import pandas as pd
import itertools
import multiprocessing
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

from numba import jit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn import metrics
from optuna.samplers import TPESampler
from joblib import Parallel, delayed

import utils


class AECloss:

    def __init__(self, C_fn, C_fp):
        self.C_fn = C_fn
        self.C_fp = C_fp

    def grad(self, y_true, y_pred):
        """
        Gradient of the loss w.r.t the elements of y_pred.

        :param y_true:
        :param y_pred:

        :return:
        """
        term_1 = -(self.C_fn * y_true * np.exp(-y_pred)) / ((1 + np.exp(-y_pred)) ** 2)
        term_2 = self.C_fp * (1 - y_true) * np.exp(-y_pred) / ((1 + np.exp(-y_pred)) ** 2)
        gradient = term_1 + term_2

        return gradient

    def hess(self, y_true, y_pred):
        """
        Second order derivative of the loss w.r.t the elements of y_pred.

        :param y_true:
        :param y_pred:

        :return:
        """

        hessian = self.C_fn * y_true * np.exp(-y_pred) / (1 + np.exp(-y_pred)) ** 2 - 2 * self.C_fn * y_true * np.exp(
            -2 * y_pred) / (1 + np.exp(-y_pred)) ** 3 \
                  - self.C_fp * (1 - y_true) * np.exp(-y_pred) / (1 + np.exp(-y_pred)) ** 2 + 2 * self.C_fp * (
                          1 - y_true) * np.exp(-2 * y_pred) / (1 + np.exp(-y_pred)) ** 3

        return hessian

    def AEC_obj(self, y_true, y_pred):
        return self.grad(y_true, y_pred), self.hess(y_true, y_pred)

    def expected_cost(self, y_true, y_pred):
        g_z = 1 / (1 + np.exp(-y_pred))
        J = y_true * ((1 - g_z) * self.C_fn) + (1 - y_true) * (g_z * self.C_fp)

        return J

    def loss(self, y_true, y_pred, C_fn=None, C_fp=None):
        if C_fn is None:
            C_fn = self.C_fn
        if C_fp is None:
            C_fp = self.C_fp
        g_z = 1 / (1 + np.exp(-y_pred))
        J = y_true * ((1 - g_z) * C_fn) + (1 - y_true) * (g_z * C_fp)
        cost = np.mean(J)

        return cost


class BiObjectiveloss:

    def __init__(self, C_fn, C_fp, theta1=1, theta2=1):
        self.C_fn = C_fn
        self.C_fp = C_fp
        self.theta1 = theta1
        self.theta2 = theta2

    def grad(self, y_true, y_pred):
        """
        Gradient of the loss w.r.t the elements of y_pred.

        :param y_true:
        :param y_pred:

        :return:
        """

        return 0.5 * self.theta1 * (
                -y_true * np.exp(-y_pred) / (1 + np.exp(-y_pred)) - (y_true - 1) * np.exp(-y_pred) / ((1 - 1 / (1
                                                                                                                + np.exp(
                    -y_pred))) * (1 + np.exp(-y_pred)) ** 2)) + 0.5 * self.theta2 * (
                -self.C_fn * y_true * np.exp(-y_pred) / (1 +
                                                         np.exp(-y_pred)) ** 2 + self.C_fp * (1 - y_true) * np.exp(
            -y_pred) / (1 + np.exp(-y_pred)) ** 2)

    def hess(self, y_true, y_pred):
        """
        Second order derivative of the loss w.r.t the elements of y_pred.

        :param y_true:
        :param y_pred:

        :return:
        """

        return 0.5 * self.theta1 * (y_true * np.exp(-y_pred) / (1 + np.exp(-y_pred)) - y_true * np.exp(-2 * y_pred) / (
                1 + np.exp(-y_pred)) ** 2 +
                                    (y_pred - 1) * np.exp(-y_pred) / (
                                            (1 - 1 / (1 + np.exp(-y_pred))) * (1 + np.exp(-y_pred)) ** 2) - 2 * (
                                            y_true - 1) * np.exp(-2 * y_pred) /
                                    ((1 - 1 / (1 + np.exp(-y_pred))) * (1 + np.exp(-y_pred)) ** 3) - (
                                            y_true - 1) * np.exp(-2 * y_pred) / (
                                            (1 - 1 / (1 + np.exp(-y_pred))) ** 2
                                            * (1 + np.exp(-y_pred)) ** 4)) + 0.5 * self.theta2 * (
                self.C_fn * y_true * np.exp(-y_pred) / (1 + np.exp(-y_pred)) ** 2 - 2 * self.C_fn * y_true * np.exp(
            -2 * y_pred) / (
                        1 + np.exp(-y_pred)) ** 3 - self.C_fp * (1 - y_true) * np.exp(-y_pred) / (
                        1 + np.exp(-y_pred)) ** 2 + 2 * self.C_fp * (1 - y_true) * np.exp(-2 * y_pred) / (
                        1 + np.exp(-y_pred)) ** 3)

    def bi_objective(self, y_true, y_pred):
        return self.grad(y_true, y_pred), self.hess(y_true, y_pred)

    def loss(self, y_true, y_pred, C_fn=None, C_fp=None):
        if C_fn is None:
            C_fn = self.C_fn
        if C_fp is None:
            C_fp = self.C_fp
        g_z = 1 / (1 + np.exp(-y_pred))
        J = y_true * ((1 - g_z) * C_fn) + (1 - y_true) * (g_z * C_fp)
        AEC = np.mean(J)

        log_loss = -np.mean(y_true * np.log(g_z) + (1 - y_true) * np.log(1 - g_z))

        obj = (self.theta1 * 0.5) * log_loss + (self.theta1 * 0.5) * AEC

        return obj


class CostInsensitive_LogisticRegression:

    def __init__(self, CV_n_iter=100, n_lambda1_grid=500, min_lambda1=1e-3, max_lambda1=1,
                 n_lambda2_grid=500, min_lambda2=1e-3, max_lambda2=1, regularization_method='none',
                 hyperparameter_optimization_method='Optuna', warm_start=True):

        self.CV_n_iter = CV_n_iter
        self.n_lambda1_grid = n_lambda1_grid
        self.min_lambda1 = min_lambda1
        self.max_lambda1 = max_lambda1
        self.n_lambda2_grid = n_lambda2_grid
        self.min_lambda2 = min_lambda2
        self.max_lambda2 = max_lambda2
        self.regularization_method = None if regularization_method == 'none' else regularization_method
        self.hyperparameter_optimization_method = hyperparameter_optimization_method
        self.warm_start = warm_start
        self.fixed_model_parameters = {'penalty': self.regularization_method,
                                       'fit_intercept': False,
                                       'solver': 'saga',
                                       'max_iter': 500}
        if self.regularization_method == 'elasticnet':
            self.fixed_model_parameters['l1_ratio'] = 0.5

        # Grid of L1/L2 penalty terms
        self._lambda1_grid = np.round(
            np.logspace(np.log10(self.min_lambda1), np.log10(self.max_lambda1), int(self.n_lambda1_grid)), 6)[::-1]
        self._lambda2_grid = np.round(np.linspace(self.min_lambda2, self.max_lambda2, num=int(self.n_lambda2_grid),
                                                  endpoint=True), 6)

        self.model = None  # Placeholder for the logistic regression model object
        self.model_hyperparameters = {}  # Placeholder for any model hyperparameters
        self.train_score = None
        self.test_score = None

        # Create the sklearn model instance so that we can use the warm start
        self.model = Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(**self.fixed_model_parameters, **{'warm_start': self.warm_start}))
        ])

    def Optuna_hyperparameter_optimization(self, X, y, t1, seed):
        """

        :param X:
        :param y:
        :param t1:

        :return:

        """

        # Construct the purged and embargoed K-fold split (without shuffling)
        CV_generator = utils.PurgedKFold(t1=t1,
                                         n_splits=5,
                                         pctEmbargo=0.025)

        # Standardization and model pipeline for GridSearchCV
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(**self.fixed_model_parameters))
        ])

        # Define the hyperparameter search space
        if self.regularization_method == 'l1' or self.regularization_method == 'l2':
            hyperarameter_search_space = {'clf__C': optuna.distributions.FloatDistribution(low=self.min_lambda1,
                                                                                           high=self.max_lambda1)}
        else:
            hyperarameter_search_space = {'clf__C': optuna.distributions.FloatDistribution(low=self.min_lambda1,
                                                                                           high=self.max_lambda1),
                                          'clf__l1_ratio': optuna.distributions.FloatDistribution(low=self.min_lambda2,
                                                                                                  high=self.max_lambda2)}

        optuna_search = optuna.integration.OptunaSearchCV(estimator=pipe,
                                                          param_distributions=hyperarameter_search_space,
                                                          cv=CV_generator.split(X=X),
                                                          n_trials=self.CV_n_iter,
                                                          n_jobs=multiprocessing.cpu_count(),
                                                          refit=False,
                                                          return_train_score=True,
                                                          scoring='neg_log_loss',
                                                          verbose=-1,
                                                          random_state=seed)

        optuna_search.fit(X=X.values,
                          y=y.values)

        # Find the best hyperparameter combinations
        best_hyperparameters = optuna_search.best_params_

        # Record the mean train/test loss function values for studying over/under-fitting.
        # Invert because sklearn maximizes the negative log loss
        best_index = optuna_search.best_index_
        train_score = optuna_search.trials_dataframe().loc[best_index, 'user_attrs_mean_train_score'] * -1
        test_score = optuna_search.best_score_ * -1

        # Rename the hyperparameters to the correct classifier attributes names
        best_hyperparameters = {key.replace('clf__', ''): value for key, value in best_hyperparameters.items()}

        return best_hyperparameters, train_score, test_score

    def fit(self, y, X, tune_hyperparameters, seed, **kwargs):
        t1 = kwargs['t1']

        # Hyperparameter optimization
        if self.regularization_method in ['l1', 'l2', 'elasticnet'] and tune_hyperparameters:
            # Hyperparameter optimization
            hyperparameter_optimization_function = getattr(self, 'Optuna_hyperparameter_optimization')

            (self.model_hyperparameters,
             self.train_score,
             self.test_score) = hyperparameter_optimization_function(X=X, y=y, t1=t1, seed=seed)

        # Define the logistic regression model.
        self.model.clf__random_state = seed
        if self.regularization_method == 'l1' or self.regularization_method == 'l2':
            self.model.clf__C = self.model_hyperparameters['C']
        else:
            self.model.clf__C = self.model_hyperparameters['C']
            self.model.clf__l1_ratio = self.model_hyperparameters['l1_ratio']

        # Fit logistic regression model
        self.model.fit(X=X.values,
                       y=y.values)

        insample_y_proba = (self.model.predict_proba(X.values)[:, 1]).flatten()

        return insample_y_proba, self.model_hyperparameters, self.train_score, self.test_score

    def predict(self, X):
        binary_y_hat = self.model.predict(X.values)[0]
        proba_y_hat = self.model.predict_proba(X.values)[0]

        return binary_y_hat, proba_y_hat


class CostSensitive_LogisticRegression:

    def __init__(self, n_jobs, CV_n_iter=100, n_lambda1_grid=100, min_lambda1=1e-3, max_lambda1=1,
                 n_lambda2_grid=100, min_lambda2=1e-3, max_lambda2=1, bi_objective=False,
                 hyperparameter_optimization_method='Optuna', regularization_method='none'):

        self.n_jobs = n_jobs
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        self.CV_n_iter = CV_n_iter
        self.n_lambda1_grid = n_lambda1_grid
        self.min_lambda1 = min_lambda1
        self.max_lambda1 = max_lambda1
        self.n_lambda2_grid = n_lambda2_grid
        self.min_lambda2 = min_lambda2
        self.max_lambda2 = max_lambda2
        self.bi_objective = bi_objective
        self.hyperparameter_optimization_method = hyperparameter_optimization_method
        self.regularization_method = None if regularization_method == 'none' else regularization_method

        # Grid of L1/L2 penalty terms
        self._lambda1_grid = np.round(
            np.logspace(np.log10(self.min_lambda1), np.log10(self.max_lambda1), int(self.n_lambda1_grid)), 6)[::-1]
        self._lambda2_grid = np.round(np.linspace(self.min_lambda2, self.max_lambda2, num=int(self.n_lambda2_grid),
                                                  endpoint=True), 6)

        self.scaler = None  # Placeholder for the full training data scaler object
        self.beta_hat = None  # Placeholder for the logistic regression beta
        self.log_loss_beta_hat = None  # Placeholder for the logistic regression (log loss objective) beta
        self.model_hyperparameters = {}  # Placeholder for any model hyperparameters
        self.train_score = None
        self.test_score = None
        self.W = np.array([3, 1])

        self.starting_value_model_parameters = {'penalty': 'l2', 'tol': 1e-6,
                                                'solver': 'saga', 'fit_intercept': False,
                                                'max_iter': 1000}

    @staticmethod
    @jit(nopython=True)
    def AEC(beta, X, y, Cfp, Cfn):
        z = X @ beta
        g_z = 1 / (1 + np.exp(-z))
        J = y * ((1 - g_z) * Cfn) + (1 - y) * (g_z * Cfp)
        cost = np.mean(J)

        return cost

    @staticmethod
    @jit(nopython=True)
    def regularized_AEC_function(beta, X, y, Cfp, Cfn, _lambda=0, alpha=1):
        z = X @ beta
        g_z = 1 / (1 + np.exp(-z))
        J = y * ((1 - g_z) * Cfn) + (1 - y) * (g_z * Cfp)

        cost = np.mean(J) + _lambda * (
                (1 - alpha) * np.linalg.norm(beta, ord=2) ** 2 / 2 + alpha * np.linalg.norm(beta, ord=1))

        return cost

    @staticmethod
    @jit(nopython=True)
    def bi_objective_loss(beta, X, y, Cfp, Cfn):
        z = X @ beta
        g_z = 1 / (1 + np.exp(-z))
        J = y * ((1 - g_z) * Cfn) + (1 - y) * (g_z * Cfp)

        cost = np.mean(J)

        log_loss = -np.mean(y * np.log(g_z) + (1 - y) * np.log(1 - g_z))

        obj = 0.5 * log_loss + 0.5 * cost

        return obj

    @staticmethod
    @jit(nopython=True)
    def bi_objectve_regularized_function(beta, X, y, Cfp, Cfn, w1, w2, _lambda=0, alpha=1):
        z = X @ beta
        g_z = 1 / (1 + np.exp(-z))
        J = y * ((1 - g_z) * Cfn) + (1 - y) * (g_z * Cfp)

        cost = np.mean(J)

        log_loss = -np.mean(y * np.log(g_z) + (1 - y) * np.log(1 - g_z))

        obj = (w1 * 0.5) * log_loss + (w2 * 0.5) * cost + _lambda * (
                (1 - alpha) * np.linalg.norm(beta, ord=2) ** 2 / 2 + alpha * np.linalg.norm(beta, ord=1))

        return obj

    @staticmethod
    @jit(nopython=True)
    def log_loss_function(beta, X, y):
        z = X @ beta
        g_z = 1 / (1 + np.exp(-z))

        obj = -np.mean(y * np.log(g_z) + (1 - y) * np.log(1 - g_z))
        if np.isnan(obj):
            return 1e10
        else:
            return obj

    @staticmethod
    @jit(nopython=True)
    def log_loss_regularized_function(beta, X, y, _lambda=0, alpha=1):
        z = X @ beta
        g_z = 1 / (1 + np.exp(-z))

        log_loss = -np.mean(y * np.log(g_z) + (1 - y) * np.log(1 - g_z))

        obj = log_loss + _lambda * (
                (1 - alpha) * np.linalg.norm(beta, ord=2) ** 2 / 2 + alpha * np.linalg.norm(beta, ord=1))

        return obj

    def find_W(self, beta0, standardized_X, y, C):

        # Find the weights for each objective (Nadir and Utopia points)
        z_star = []
        z_N = []

        log_loss_optim_args = (standardized_X, y.values)
        AEC_optim_args = (standardized_X, y.values, C['fp'].values, C['fn'].values)

        # Minimize the regularized log loss objective at the model hyperparameters
        log_loss_optim = minimize(fun=CostSensitive_LogisticRegression.log_loss_function,
                                  x0=beta0,
                                  args=log_loss_optim_args,
                                  method='SLSQP',
                                  options={'maxiter': 1000})
        log_loss_beta_hat = log_loss_optim['x']
        y_pred_LR = 1 / (1 + np.exp(-(standardized_X @ log_loss_beta_hat)))

        z_star.append(metrics.log_loss(y_true=y.values,
                                       y_pred=y_pred_LR.flatten()))

        AEC_optim = minimize(fun=CostSensitive_LogisticRegression.AEC,
                             x0=beta0,
                             args=AEC_optim_args,
                             method='SLSQP',
                             options={'maxiter': 1000})
        if not AEC_optim.success:
            AEC_optim = minimize(fun=CostSensitive_LogisticRegression.AEC,
                                 x0=self.beta_hat,
                                 args=AEC_optim_args,
                                 method='SLSQP',
                                 options={'maxiter': 1000})
        AEC_beta_hat = AEC_optim['x']

        y_pred_AEC = 1 / (1 + np.exp(-(standardized_X @ AEC_beta_hat)))
        z_star.append(CostSensitive_LogisticRegression.AEC(beta=AEC_beta_hat,
                                                           X=standardized_X, y=y.values,
                                                           Cfp=C['fp'].values, Cfn=C['fn'].values))

        z_N.append(max([metrics.log_loss(y_true=y.values,
                                         y_pred=y_pred_LR.flatten()),
                        metrics.log_loss(y_true=y.values,
                                         y_pred=y_pred_AEC.flatten())
                        ]))

        z_N.append(max([CostSensitive_LogisticRegression.AEC(beta=log_loss_beta_hat,
                                                             X=standardized_X, y=y.values,
                                                             Cfp=C['fp'].values, Cfn=C['fn'].values),
                        CostSensitive_LogisticRegression.AEC(beta=AEC_beta_hat,
                                                             X=standardized_X, y=y.values,
                                                             Cfp=C['fp'].values, Cfn=C['fn'].values)
                        ]))

        if np.min((np.array(z_N) - np.array(z_star))) <= 1e-6:
            W = np.array(z_N) - np.array(z_star)
            W[W <= 1e-6] = 1
            W = 1 / W
        else:
            W = 1 / (np.array(z_N) - np.array(z_star))

        return W


    @staticmethod
    def _optuna_objective_CV_loop(train_X, test_X, train_y, test_y, starting_value_model_parameters,
                                  regularization_method, _lambda, alpha, beta_hat):

        # Standardize X
        X_scaler = StandardScaler()
        standardized_train_X = X_scaler.fit_transform(train_X)
        standardized_test_X = X_scaler.transform(test_X)

        # Get starting values for numerical optimization
        starting_values_LR_model = LogisticRegression(**starting_value_model_parameters)
        starting_values_LR_model.fit(X=standardized_train_X,
                                     y=train_y.values)
        beta0 = starting_values_LR_model.coef_.T.flatten()

        # Get arguments for optimization
        if regularization_method == 'l1':
            args = (standardized_train_X, train_y.values, _lambda)
        else:
            args = (standardized_train_X, train_y.values, _lambda, alpha)

        optim_result = minimize(fun=CostSensitive_LogisticRegression.log_loss_regularized_function,
                                x0=beta_hat,
                                args=args,
                                method='SLSQP',
                                options={'maxiter': 500})
        if not optim_result.success:
            optim_result = minimize(fun=CostSensitive_LogisticRegression.log_loss_regularized_function,
                                    x0=beta0,
                                    args=args,
                                    method='SLSQP',
                                    options={'maxiter': 500})

            if optim_result.success:
                fold_beta_hat = optim_result['x']
                fold_y_proba_hat = 1 / (1 + np.exp(-(standardized_test_X @ fold_beta_hat)))
                loss = metrics.log_loss(y_true=test_y.values,
                                        y_pred=fold_y_proba_hat)
            else:
                loss = 1e10
        else:
            fold_beta_hat = optim_result['x']
            fold_y_proba_hat = 1 / (1 + np.exp(-(standardized_test_X @ fold_beta_hat)))
            loss = metrics.log_loss(y_true=test_y.values,
                                    y_pred=fold_y_proba_hat)
        return loss

    @staticmethod
    def _optuna_objective(trial, X, y, t1, beta_hat, regularization_method, min_lambda1, max_lambda1,
                          min_lambda2, max_lambda2, starting_value_model_parameters):

        # Construct the purged and embargoed K-fold split (without shuffling)
        CV_generator = utils.PurgedKFold(t1=t1,
                                         n_splits=5,
                                         pctEmbargo=0.025)

        if regularization_method == 'l1':
            _lambda = trial.suggest_float(name='lambda',
                                          low=min_lambda1,
                                          high=max_lambda1,
                                          log=True)
            alpha = None
        else:
            _lambda = trial.suggest_float(name='lambda',
                                          low=min_lambda1,
                                          high=max_lambda1,
                                          log=True)
            alpha = trial.suggest_float(name='alpha',
                                        low=min_lambda2,
                                        high=max_lambda2,
                                        log=False)

        loss = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(CostSensitive_LogisticRegression._optuna_objective_CV_loop)(X.iloc[train_idx, :],
                                                                                X.iloc[test_idx, :],
                                                                                y.iloc[train_idx],
                                                                                y.iloc[test_idx],
                                                                                starting_value_model_parameters,
                                                                                regularization_method,
                                                                                _lambda,
                                                                                alpha,
                                                                                beta_hat)
            for train_idx, test_idx in CV_generator.split(X))

        # Compute the mean loss over all test folds
        score = np.nanmean(loss)

        return score

    def Optuna_hyperparameter_optimization(self, X, y, t1, seed):
        """

        :param X:
        :param y:
        :param t1:

        :return:

        """

        optuna_objective = lambda trial: CostSensitive_LogisticRegression._optuna_objective(trial,
                                                                                            X=X,
                                                                                            y=y,
                                                                                            t1=t1,
                                                                                            beta_hat=self.beta_hat,
                                                                                            regularization_method=self.regularization_method,
                                                                                            min_lambda1=self.min_lambda1,
                                                                                            max_lambda1=self.max_lambda1,
                                                                                            min_lambda2=self.min_lambda2,
                                                                                            max_lambda2=self.max_lambda2,
                                                                                            starting_value_model_parameters=self.starting_value_model_parameters)

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler, direction='minimize')
        study.optimize(optuna_objective,
                       n_trials=self.CV_n_iter,
                       n_jobs=multiprocessing.cpu_count())

        # Find the best hyperparameter combinations
        best_hyperparameters = study.best_params

        # Record the mean train/test loss function values for studying over/under-fitting.
        # Invert because sklearn maximizes the negative log loss
        train_score = np.nan
        test_score = study.best_value

        return best_hyperparameters, train_score, test_score

    def fit(self, X, y, tune_hyperparameters, seed, **kwargs):

        C = kwargs['C']
        t1 = kwargs['t1']

        self.starting_value_model_parameters['random_state'] = seed

        # Standardize X
        self.X_scaler = StandardScaler()
        standardized_X = self.X_scaler.fit_transform(X)
        # Add intercept
        # standardized_X = np.concatenate([np.ones((len(standardized_X), 1)), standardized_X], axis=1)

        # Take the cost insensitive parameters as staring values
        cost_insensitive_LR_model = LogisticRegression(**self.starting_value_model_parameters)
        cost_insensitive_LR_model.fit(X=standardized_X,
                                      y=y.values)
        beta0 = cost_insensitive_LR_model.coef_.T.flatten()

        if self.beta_hat is None:  # Set beta hat in the first iteration to cost insensitive ridge regression parameters
            self.beta_hat = beta0

        # Hyperparameter optimization
        if self.regularization_method in ['l1', 'elasticnet'] and tune_hyperparameters:
            # Hyperparameter optimization
            hyperparameter_optimization_function = getattr(self, 'Optuna_hyperparameter_optimization')

            (self.model_hyperparameters,
             self.train_score,
             self.test_score) = hyperparameter_optimization_function(X=X,
                                                                     y=y,
                                                                     t1=t1,
                                                                     seed=seed)

        if self.bi_objective:
            W = self.find_W(beta0=beta0,
                            standardized_X=standardized_X,
                            y=y,
                            C=C)
            if np.any(np.isnan(W)):
                # Use the previous iteration weights when we cannot correctly determine this period's weights
                pass
            else:
                self.W = W

        if self.bi_objective:
            # Find model parameters (beta) via numerical optimization
            if self.regularization_method == 'l1':
                _lambda = self.model_hyperparameters['lambda']
                args = (standardized_X, y.values, C['fp'].values, C['fn'].values, self.W[0], self.W[1],
                        _lambda)
            elif self.regularization_method == 'elasticnet':
                _lambda = self.model_hyperparameters['lambda']
                alpha = self.model_hyperparameters['alpha']
                args = (standardized_X, y.values, C['fp'].values, C['fn'].values, self.W[0], self.W[1],
                        _lambda,
                        alpha)
            else:
                args = (standardized_X, y.values, C['fp'].values, C['fn'].values, self.W[0], self.W[1])

            optim_result = minimize(fun=CostSensitive_LogisticRegression.bi_objectve_regularized_function,
                                    x0=beta0,
                                    args=args,
                                    method='SLSQP',
                                    options={'maxiter': 1000})
            if not optim_result.success:
                optim_result = minimize(fun=CostSensitive_LogisticRegression.bi_objectve_regularized_function,
                                        x0=self.beta_hat,
                                        args=args,
                                        method='SLSQP',
                                        options={'maxiter': 1000})

            self.beta_hat = optim_result['x']

        else:
            if self.regularization_method == 'l1':
                _lambda = self.model_hyperparameters['lambda']
                args = (standardized_X, y.values, C['fp'].values, C['fn'].values, _lambda)
            elif self.regularization_method == 'elasticnet':
                _lambda = self.model_hyperparameters['lambda']
                alpha = self.model_hyperparameters['alpha']
                args = (standardized_X, y.values, C['fp'].values, C['fn'].values, _lambda, alpha)
            else:
                args = (standardized_X, y.values, C['fp'].values, C['fn'].values)

            # Find model parameters (beta) via numerical optimization
            optim_result = minimize(fun=CostSensitive_LogisticRegression.regularized_AEC_function,
                                    x0=beta0,
                                    args=args,
                                    method='SLSQP',
                                    options={'maxiter': 1000})

            if not optim_result.success:
                optim_result = minimize(fun=CostSensitive_LogisticRegression.regularized_AEC_function,
                                        x0=self.beta_hat,
                                        args=args,
                                        method='SLSQP',
                                        options={'maxiter': 1000})
            self.beta_hat = optim_result['x']

        insample_z = standardized_X @ self.beta_hat
        insample_y_proba = (1 / (1 + np.exp(-insample_z))).flatten()

        return insample_y_proba, self.model_hyperparameters, self.train_score, self.test_score

    def predict(self, X):

        # Standardize X
        standardized_X = self.X_scaler.transform(X)

        # Add intercept
        # standardized_X = np.concatenate([np.ones((len(standardized_X), 1)), standardized_X], axis=1)

        z = standardized_X @ self.beta_hat
        p_hat = 1 / (1 + np.exp(-z))[0]
        proba_y_hat = np.array([1 - p_hat, p_hat])
        binary_y_hat = (p_hat >= 0.5).astype(int)

        return binary_y_hat, proba_y_hat


class GradientBoostingMachine_model:

    def __init__(self, n_estimators_search_space, num_leaves_search_space, max_depth_search_space,
                 learning_rate_search_space,
                 subsample_search_space, min_child_samples_search_space, max_bin_search_space, CV_n_iter,
                 n_lambda1_grid=None, min_lambda1=None, max_lambda1=None, n_lambda2_grid=None, min_lambda2=None,
                 max_lambda2=None, hyperparameter_optimization_method='Optuna'):

        self.n_estimators_search_space = n_estimators_search_space
        self.num_leaves_search_space = num_leaves_search_space
        self.max_depth_search_space = max_depth_search_space
        self.learning_rate_search_space = learning_rate_search_space
        self.subsample_search_space = subsample_search_space
        self.min_child_samples_search_space = min_child_samples_search_space
        self.max_bin_search_space = max_bin_search_space
        self.n_lambda1_grid = n_lambda1_grid
        self.min_lambda1 = min_lambda1
        self.max_lambda1 = max_lambda1
        self.n_lambda2_grid = n_lambda2_grid
        self.min_lambda2 = min_lambda2
        self.max_lambda2 = max_lambda2
        self.hyperparameter_optimization_method = hyperparameter_optimization_method

        self.CV_n_iter = CV_n_iter

        if self.n_lambda1_grid is not None:
            self._lambda1_grid = np.round(
                np.logspace(np.log10(self.min_lambda1), np.log10(self.max_lambda1), int(self.n_lambda1_grid)), 6)[::-1]
        else:
            self._lambda1_grid = [0.0]
        if self.n_lambda2_grid is not None:
            self._lambda2_grid = np.round(
                np.logspace(np.log10(self.min_lambda2), np.log10(self.max_lambda2), int(self.n_lambda2_grid)), 6)[::-1]
        else:
            self._lambda2_grid = [0.0]

        self.fixed_base_model_parameters = {'boosting_type': 'dart',
                                            'objective': 'binary',
                                            'n_jobs': -1,
                                            'subsample_freq': 1,
                                            'extra_trees': False,
                                            'verbosity': -1}

        self.model = None
        self.model_hyperparameters = {}
        self.train_score = None
        self.test_score = None

        self.starting_value_model_parameters = {'penalty': 'l2', 'tol': 1e-6,
                                                'solver': 'saga', 'fit_intercept': False,
                                                'max_iter': 1000}

    @staticmethod
    def _optuna_objective(trial, X, y, t1, seed, n_estimators_search_space, num_leaves_search_space,
                          max_depth_search_space, learning_rate_search_space, subsample_search_space,
                          min_child_samples_search_space, max_bin_search_space, min_lambda1, max_lambda1,
                          min_lambda2, max_lambda2, fixed_base_model_parameters, starting_value_model_parameters):

        # Construct the purged and embargoed K-fold split (without shuffling)
        CV_generator = utils.PurgedKFold(t1=t1,
                                         n_splits=5,
                                         pctEmbargo=0.025)

        loss = []  # Test set loss using beta0 starting values

        # Define the hyperparameter search space
        n_estimators = trial.suggest_int(name='n_estimators',
                                         low=np.min(n_estimators_search_space),
                                         high=np.max(n_estimators_search_space),
                                         step=10)
        num_leaves = trial.suggest_int(name='num_leaves',
                                       low=np.min(num_leaves_search_space),
                                       high=np.max(num_leaves_search_space),
                                       step=8)
        max_depth = trial.suggest_int(name='max_depth',
                                      low=np.min(max_depth_search_space),
                                      high=np.max(max_depth_search_space))
        learning_rate = trial.suggest_float(name='learning_rate',
                                            low=np.min(learning_rate_search_space),
                                            high=np.max(learning_rate_search_space)),
        subsample = trial.suggest_float(name='subsample',
                                        low=np.min(subsample_search_space),
                                        high=np.max(subsample_search_space),
                                        step=0.01)
        min_child_samples = trial.suggest_int(name='min_child_samples',
                                              low=np.min(min_child_samples_search_space),
                                              high=np.max(min_child_samples_search_space),
                                              step=10),
        max_bin = trial.suggest_int(name='max_bin',
                                    low=np.min(max_bin_search_space),
                                    high=np.max(max_bin_search_space),
                                    step=5)

        reg_alpha = trial.suggest_float(name='reg_alpha',
                                        low=min_lambda1,
                                        high=max_lambda1,
                                        log=True)
        reg_lambda = trial.suggest_float(name='reg_lambda',
                                         low=min_lambda2,
                                         high=max_lambda2,
                                         log=False)

        LGBM_mdl = lgb.LGBMClassifier(**fixed_base_model_parameters,
                                      n_estimators=n_estimators,
                                      num_leaves=num_leaves,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      subsample=subsample,
                                      min_child_samples=min_child_samples,
                                      max_bin=max_bin,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      random_state=seed)

        for train_idx, test_idx in CV_generator.split(X):
            train_X, test_X = X.iloc[train_idx, :], X.iloc[test_idx, :]
            train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]

            # Standardize X
            X_scaler = StandardScaler()
            standardized_train_X = X_scaler.fit_transform(train_X)
            standardized_test_X = X_scaler.transform(test_X)

            # Get starting values for numerical optimization
            starting_values_LR_model = LogisticRegression(**starting_value_model_parameters)
            starting_values_LR_model.fit(X=standardized_train_X,
                                         y=train_y.values)
            init_score = starting_values_LR_model.predict_proba(X=standardized_train_X)[:, 1]

            # Fit GBM
            LGBM_mdl.fit(X=standardized_train_X,
                         y=train_y.values,
                         init_score=init_score)
            fold_y_proba_hat = LGBM_mdl.predict_proba(X=standardized_test_X,
                                                      num_iteration=-1)[:, 1]
            loss.append(metrics.log_loss(y_true=test_y.values,
                                         y_pred=fold_y_proba_hat))

        # Compute the mean loss over all test folds
        score = np.nanmean(loss)

        return score

    def Optuna_hyperparameter_optimization(self, X, y, t1, seed):
        """

        :param X:
        :param y:
        :param t1:

        :return:

        """

        optuna_objective = lambda trial: GradientBoostingMachine_model._optuna_objective(trial,
                                                                                         X=X,
                                                                                         y=y,
                                                                                         t1=t1,
                                                                                         seed=seed,
                                                                                         n_estimators_search_space=self.n_estimators_search_space,
                                                                                         num_leaves_search_space=self.num_leaves_search_space,
                                                                                         max_depth_search_space=self.max_depth_search_space,
                                                                                         learning_rate_search_space=self.learning_rate_search_space,
                                                                                         subsample_search_space=self.subsample_search_space,
                                                                                         min_child_samples_search_space=self.min_child_samples_search_space,
                                                                                         max_bin_search_space=self.max_bin_search_space,
                                                                                         min_lambda1=self.min_lambda1,
                                                                                         max_lambda1=self.max_lambda1,
                                                                                         min_lambda2=self.min_lambda2,
                                                                                         max_lambda2=self.max_lambda2,
                                                                                         fixed_base_model_parameters=self.fixed_base_model_parameters,
                                                                                         starting_value_model_parameters=self.starting_value_model_parameters)

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler, direction='minimize')
        study.optimize(optuna_objective,
                       n_trials=self.CV_n_iter,
                       n_jobs=multiprocessing.cpu_count())

        # Find the best hyperparameter combinations
        best_hyperparameters = study.best_params

        # Record the mean train/test loss function values for studying over/under-fitting.
        # Invert because sklearn maximizes the negative log loss
        train_score = np.nan
        test_score = study.best_value

        return best_hyperparameters, train_score, test_score

    def fit(self, X, y, t1, tune_hyperparameters, seed):

        # Set the feature (column) subsample ratio hyperparameter to sqrt rule of thumb
        self.fixed_base_model_parameters['colsample_bytree'] = np.sqrt(X.shape[1]) / X.shape[1]

        if tune_hyperparameters:
            # Hyperparameter optimization
            hyperparameter_optimization_function = getattr(self, 'Optuna_hyperparameter_optimization')

            (self.model_hyperparameters,
             self.train_score,
             self.test_score) = hyperparameter_optimization_function(X=X, y=y, t1=t1, seed=seed)

        # Define the uncalibrated GBM regression model.
        model_params = {**self.fixed_base_model_parameters, **self.model_hyperparameters, **{'random_state': seed}}

        self.model = Pipeline([
            ('scale', StandardScaler()),
            ('clf', lgb.LGBMClassifier(**model_params))
        ])

        # initial Logisitc regression fit for the init_score parameter
        _LR_model = Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(**self.starting_value_model_parameters))
        ])
        _LR_model.fit(X=X.values,
                      y=y.values)
        init_score = _LR_model.predict_proba(X=X.values)[:, 1]

        self.model.fit(X=X.values,
                       y=y.values,
                       **{'clf__init_score': init_score})

        insample_y_proba = self.model.predict_proba(X=X.values, num_iteration=-1)[:, 1].flatten()

        return insample_y_proba, self.model_hyperparameters, self.train_score, self.test_score

    def predict(self, X):

        proba_y_hat = self.model.predict_proba(X=X.values,
                                               num_iteration=-1)
        binary_y_hat = self.model.predict(X=X.values,
                                          num_iteration=-1)[0]

        return binary_y_hat, proba_y_hat


class CostSensitive_GradientBoostingMachine_model:

    def __init__(self, n_estimators_search_space, num_leaves_search_space, max_depth_search_space,
                 learning_rate_search_space,
                 subsample_search_space, min_child_samples_search_space, max_bin_search_space, CV_n_iter, n_jobs=-1,
                 bi_objective=False, n_lambda1_grid=None, min_lambda1=None, max_lambda1=None, n_lambda2_grid=None,
                 min_lambda2=None, max_lambda2=None, hyperparameter_optimization_method='Optuna'):

        self.n_estimators_search_space = n_estimators_search_space
        self.num_leaves_search_space = num_leaves_search_space
        self.max_depth_search_space = max_depth_search_space
        self.learning_rate_search_space = learning_rate_search_space
        self.subsample_search_space = subsample_search_space
        self.min_child_samples_search_space = min_child_samples_search_space
        self.max_bin_search_space = max_bin_search_space
        self.n_lambda1_grid = n_lambda1_grid
        self.min_lambda1 = min_lambda1
        self.max_lambda1 = max_lambda1
        self.n_lambda2_grid = n_lambda2_grid
        self.min_lambda2 = min_lambda2
        self.max_lambda2 = max_lambda2
        self.hyperparameter_optimization_method = hyperparameter_optimization_method

        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.bi_objective = bi_objective
        self.CV_n_iter = CV_n_iter

        if self.n_lambda1_grid is not None:
            self._lambda1_grid = np.round(
                np.logspace(np.log10(self.min_lambda1), np.log10(self.max_lambda1), int(self.n_lambda1_grid)), 6)[::-1]
        else:
            self._lambda1_grid = [0.0]
        if self.n_lambda2_grid is not None:
            self._lambda2_grid = np.round(np.linspace(self.min_lambda2, self.max_lambda2, num=int(self.n_lambda2_grid),
                                                      endpoint=True), 2)
        else:
            self._lambda2_grid = [0.0]

        self.fixed_base_model_parameters = {'boosting_type': 'dart',
                                            'subsample_freq': 1,
                                            'extra_trees': False,
                                            'verbosity': -1}
        self.model = None
        self.model_hyperparameters = {}
        self.train_score = None
        self.test_score = None
        self.W = np.array([3, 1])

        self.starting_value_model_parameters = {'penalty': 'l2', 'tol': 1e-6,
                                                'solver': 'saga', 'fit_intercept': False,
                                                'max_iter': 1000}

    def find_W(self, init_score, standardized_X, y, C):

        # Find the weights for each objective (Nadir and Utopia points)
        z_star = []
        z_N = []

        # Minimize the log loss objective at the model hyperparameters
        log_loss_obj_LGBM_model_parameters = {**self.fixed_base_model_parameters, **self.model_hyperparameters,
                                              **{'objective': 'binary',
                                                 'n_jobs': -1}}
        log_loss_obj_LGBM_model = lgb.LGBMClassifier(**log_loss_obj_LGBM_model_parameters)
        log_loss_obj_LGBM_model.fit(X=standardized_X,
                                    y=y.values,
                                    init_score=init_score)
        log_loss_obj_LGBM_model_y_pred_raw = log_loss_obj_LGBM_model.predict(X=standardized_X,
                                                                             num_iteration=-1,
                                                                             raw_score=True)
        log_loss_obj_LGBM_model_y_proba_hat = 1 / (1 + np.exp(-log_loss_obj_LGBM_model_y_pred_raw))
        z_star.append(metrics.log_loss(y_true=y.values,
                                       y_pred=log_loss_obj_LGBM_model_y_proba_hat.flatten()))

        AEC_obj = AECloss(C_fn=C['fn'].values,
                          C_fp=C['fp'].values)
        AEC_obj_LGBM_model_parameters = {**self.fixed_base_model_parameters, **self.model_hyperparameters,
                                         **{'objective': AEC_obj.AEC_obj,
                                            'n_jobs': -1}}
        AEC_obj_LGBM_model = lgb.LGBMClassifier(**AEC_obj_LGBM_model_parameters)
        AEC_obj_LGBM_model.fit(X=standardized_X,
                               y=y.values,
                               init_score=init_score)
        AEC_obj_LGBM_model_y_pred_raw = AEC_obj_LGBM_model.predict(X=standardized_X,
                                                                   num_iteration=-1,
                                                                   raw_score=True)
        AEC_obj_LGBM_model_y_proba_hat = 1 / (1 + np.exp(-AEC_obj_LGBM_model_y_pred_raw))

        z_star.append(AEC_obj.loss(y_true=y.values,
                                   y_pred=AEC_obj_LGBM_model_y_proba_hat))

        z_N.append(max([metrics.log_loss(y_true=y.values,
                                         y_pred=log_loss_obj_LGBM_model_y_proba_hat.flatten()),
                        metrics.log_loss(y_true=y.values,
                                         y_pred=AEC_obj_LGBM_model_y_proba_hat.flatten())
                        ]))
        z_N.append(max([AEC_obj.loss(y_true=y.values,
                                     y_pred=log_loss_obj_LGBM_model_y_pred_raw),
                        AEC_obj.loss(y_true=y.values,
                                     y_pred=AEC_obj_LGBM_model_y_pred_raw)
                        ]))
        if np.min((np.array(z_N) - np.array(z_star))) <= 1e-6:
            W = np.array(z_N) - np.array(z_star)
            W[W <= 1e-6] = 1
            W = 1 / W
        else:
            W = 1 / (np.array(z_N) - np.array(z_star))

        return W

    @staticmethod
    def _optuna_objective_CV_loop(train_X, test_X, train_y, test_y, starting_value_model_parameters, fixed_base_model_parameters,
                                  n_estimators, num_leaves, max_depth, learning_rate, subsample, min_child_samples, max_bin,
                                  reg_alpha, reg_lambda, seed):

        LGBM_mdl = lgb.LGBMClassifier(**fixed_base_model_parameters,
                                      n_estimators=n_estimators,
                                      num_leaves=num_leaves,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      subsample=subsample,
                                      min_child_samples=min_child_samples,
                                      max_bin=max_bin,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      random_state=seed,
                                      n_jobs=-1)

        # Standardize X
        X_scaler = StandardScaler()
        standardized_train_X = X_scaler.fit_transform(train_X)
        standardized_test_X = X_scaler.transform(test_X)

        # Get starting values for numerical optimization
        starting_values_LR_model = LogisticRegression(**starting_value_model_parameters)
        starting_values_LR_model.fit(X=standardized_train_X,
                                     y=train_y.values)
        init_score = starting_values_LR_model.predict_proba(X=standardized_train_X)[:, 1]

        # Fit GBM
        LGBM_mdl.fit(X=standardized_train_X,
                     y=train_y.values,
                     init_score=init_score)
        fold_y_proba_hat = LGBM_mdl.predict_proba(X=standardized_test_X,
                                                  num_iteration=-1)[:, 1]
        loss = metrics.log_loss(y_true=test_y.values,
                                     y_pred=fold_y_proba_hat)

        return loss


    @staticmethod
    def _optuna_objective(trial, X, y, t1, seed, n_estimators_search_space, num_leaves_search_space,
                          max_depth_search_space, learning_rate_search_space, subsample_search_space,
                          min_child_samples_search_space, max_bin_search_space, min_lambda1, max_lambda1,
                          min_lambda2, max_lambda2, fixed_base_model_parameters, starting_value_model_parameters):

        # Construct the purged and embargoed K-fold split (without shuffling)
        CV_generator = utils.PurgedKFold(t1=t1,
                                         n_splits=5,
                                         pctEmbargo=0.025)

        loss = []  # Test set loss using beta0 starting values
        # Define the hyperparameter search space
        n_estimators = trial.suggest_int(name='n_estimators',
                                         low=np.min(n_estimators_search_space),
                                         high=np.max(n_estimators_search_space),
                                         step=10)
        num_leaves = trial.suggest_int(name='num_leaves',
                                       low=np.min(num_leaves_search_space),
                                       high=np.max(num_leaves_search_space),
                                       step=8)
        max_depth = trial.suggest_int(name='max_depth',
                                      low=np.min(max_depth_search_space),
                                      high=np.max(max_depth_search_space))
        learning_rate = trial.suggest_float(name='learning_rate',
                                            low=np.min(learning_rate_search_space),
                                            high=np.max(learning_rate_search_space)),
        subsample = trial.suggest_float(name='subsample',
                                        low=np.min(subsample_search_space),
                                        high=np.max(subsample_search_space),
                                        step=0.01)
        min_child_samples = trial.suggest_int(name='min_child_samples',
                                              low=np.min(min_child_samples_search_space),
                                              high=np.max(min_child_samples_search_space),
                                              step=10),
        max_bin = trial.suggest_int(name='max_bin',
                                    low=np.min(max_bin_search_space),
                                    high=np.max(max_bin_search_space),
                                    step=5)

        reg_alpha = trial.suggest_float(name='reg_alpha',
                                        low=min_lambda1,
                                        high=max_lambda1,
                                        log=True)
        reg_lambda = trial.suggest_float(name='reg_lambda',
                                         low=min_lambda2,
                                         high=max_lambda2,
                                         log=False)


        loss = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(CostSensitive_LogisticRegression._optuna_objective_CV_loop)(X.iloc[train_idx, :],
                                                                                X.iloc[test_idx, :],
                                                                                y.iloc[train_idx],
                                                                                y.iloc[test_idx],
                                                                                starting_value_model_parameters,
                                                                                fixed_base_model_parameters,
                                                                                n_estimators, num_leaves, max_depth,
                                                                                learning_rate, subsample,
                                                                                min_child_samples, max_bin,
                                                                                reg_alpha, reg_lambda, seed)
            for train_idx, test_idx in CV_generator.split(X))

        # Compute the mean loss over all test folds
        score = np.nanmean(loss)

        return score

    def Optuna_hyperparameter_optimization(self, X, y, t1, seed):
        """

        :param X:
        :param y:
        :param t1:

        :return:

        """

        optuna_objective = lambda trial: GradientBoostingMachine_model._optuna_objective(trial,
                                                                                         X=X,
                                                                                         y=y,
                                                                                         t1=t1,
                                                                                         seed=seed,
                                                                                         n_estimators_search_space=self.n_estimators_search_space,
                                                                                         num_leaves_search_space=self.num_leaves_search_space,
                                                                                         max_depth_search_space=self.max_depth_search_space,
                                                                                         learning_rate_search_space=self.learning_rate_search_space,
                                                                                         subsample_search_space=self.subsample_search_space,
                                                                                         min_child_samples_search_space=self.min_child_samples_search_space,
                                                                                         max_bin_search_space=self.max_bin_search_space,
                                                                                         min_lambda1=self.min_lambda1,
                                                                                         max_lambda1=self.max_lambda1,
                                                                                         min_lambda2=self.min_lambda2,
                                                                                         max_lambda2=self.max_lambda2,
                                                                                         fixed_base_model_parameters=self.fixed_base_model_parameters,
                                                                                         starting_value_model_parameters=self.starting_value_model_parameters)

        sampler = TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler, direction='minimize')
        study.optimize(optuna_objective,
                       n_trials=self.CV_n_iter,
                       n_jobs=multiprocessing.cpu_count())

        # Find the best hyperparameter combinations
        best_hyperparameters = study.best_params

        # Record the mean train/test loss function values for studying over/under-fitting.
        # Invert because sklearn maximizes the negative log loss
        train_score = np.nan
        test_score = study.best_value

        return best_hyperparameters, train_score, test_score

    def fit(self, X, y, t1, C, tune_hyperparameters, seed):

        # Set the feature (column) subsample ratio hyperparameter to sqrt rule of thumb
        self.fixed_base_model_parameters['colsample_bytree'] = np.sqrt(X.shape[1]) / X.shape[1]

        if tune_hyperparameters:
            # Hyperparameter optimization
            if self.hyperparameter_optimization_method == 'RandomizedSearchCV':
                hyperparameter_optimization_function = getattr(self, 'RandomizedSearchCV_hyperparameter_optimization')
            elif self.hyperparameter_optimization_method == 'Optuna':
                hyperparameter_optimization_function = getattr(self, 'Optuna_hyperparameter_optimization')

            (self.model_hyperparameters,
             self.train_score,
             self.test_score) = hyperparameter_optimization_function(X=X,
                                                                     y=y,
                                                                     t1=t1,
                                                                     seed=seed)

        # Standardize X
        self.X_scaler = StandardScaler()
        standardized_X = self.X_scaler.fit_transform(X)

        # Get starting values for numerical optimization
        starting_values_LR_model = LogisticRegression(**self.starting_value_model_parameters)
        starting_values_LR_model.fit(X=standardized_X,
                                     y=y.values)
        init_score = starting_values_LR_model.predict_proba(X=standardized_X)[:, 1]

        if self.bi_objective:

            W = self.find_W(init_score=init_score,
                            standardized_X=standardized_X,
                            y=y,
                            C=C)

            if np.any(np.isnan(W)):
                # Use the previous iteration weights when we cannot correctly determine this period's weights
                pass
            else:
                self.W = W

            _bi_objective = BiObjectiveloss(C_fn=C['fn'].values,
                                            C_fp=C['fp'].values,
                                            theta1=self.W[0],
                                            theta2=self.W[1])
            LGBM_bi_objective_model_parameters = {**self.fixed_base_model_parameters, **self.model_hyperparameters,
                                                  **{'objective': _bi_objective.bi_objective,
                                                     'n_jobs': -1,
                                                     'random_state': seed}}

            self.model = lgb.LGBMClassifier(**LGBM_bi_objective_model_parameters)

            self.model.fit(X=standardized_X,
                           y=y.values,
                           init_score=init_score)

        else:
            # Define the GBM regression model.
            # AEC class instance
            AEC_obj = AECloss(C_fn=C['fn'].values,
                              C_fp=C['fp'].values)
            model_params = {**self.fixed_base_model_parameters, **self.model_hyperparameters,
                            **{'objective': AEC_obj.AEC_obj,
                               'n_jobs': -1,
                               'random_state': seed
                               }
                            }

            self.model = lgb.LGBMClassifier(**model_params)

            self.model.fit(X=standardized_X,
                           y=y.values,
                           init_score=init_score)

        z = self.model.predict(X=standardized_X,
                               raw_score=True,
                               num_iteration=-1)
        insample_p_hat = (1 / (1 + np.exp(-z))).flatten()

        return insample_p_hat, self.model_hyperparameters, self.train_score, self.test_score

    def predict(self, X):

        # Standardize X
        standardized_X = self.X_scaler.transform(X)

        z = self.model.predict(X=standardized_X,
                               raw_score=True,
                               num_iteration=-1)
        p_hat = 1 / (1 + np.exp(-z))[0]
        proba_y_hat = np.array([1 - p_hat, p_hat])
        binary_y_hat = (p_hat >= 0.5).astype(int)

        return binary_y_hat, proba_y_hat
