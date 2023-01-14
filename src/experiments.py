"""Implements experiments.

Implementation of `Experiment` base  class, as well as derived experiments.
"""
import abc
import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np

from src import losses
from src.data import synthetic_data
from src.models import SGDSIPDeconvolution1D
from src.utils.paths import PROJECT_ROOT, InPath
from src.utils.logs import FORMATTER


class Experiment(abc.ABC):
    """Base experiment class."""
    def __init__(self):
        self.logger = logging.getLogger("main.experiment")
        self.has_model = False
        self.params = None # to be set after a model is made

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def get_experiment_params_dict(self) -> dict:
        pass

    @abc.abstractmethod
    def make_model(self):
        pass

    def save_params_dict(self):
        assert self.params is not None
        self.logger.info("Saving parameters.")
        with open("params.pkl", "wb") as file:
            pickle.dump(self.params, file)

    def save_model(self):
        self.logger.info("Saving fitted model.")
        with open("model.pkl", "wb") as file:
            pickle.dump(self.model, file)

    def find_fitted_model(self) -> Union[Path, None]:
        """Scans experiment folder for already fitted model, with the same
        parameters as current instance. Returns model path if found, else
        returns None.

        """
        self.logger.info("Searching for saved fitted model.")
        my_params = self.get_experiment_params_dict()
        for child in self.path.iterdir():
            param_path = child / "params.pkl"
            model_path = child / "model.pkl"
            if param_path.exists() and model_path.exists():
                with open(param_path, "rb") as file:
                    saved_params = pickle.load(file)
                if my_params == saved_params:
                    self.logger.info("Found saved fitted model.")
                    return model_path
        self.logger.info("Saved fitted model not found.")
        return None


    def setup_run(self) -> Path:
        """Some boilerplate code to setup an experiment's run."""
        run_path = self.path / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.logger.info("Creating run folder.")
        run_path.mkdir(exist_ok=True)
        fh = logging.FileHandler(run_path / "run.log", mode="a")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(FORMATTER)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        return run_path


class ArticleDeconvolutionExample(Experiment):
    """This experiment is aimed at recreating the the article's plots."""
    path = PROJECT_ROOT / "runs" / "article_deconvolution_example"
    path.mkdir(parents=True, exist_ok=True)
    def __init__(self):
        super().__init__()


    def get_experiment_params_dict(self) -> dict:
        assert self.has_model
        params = self.model.get_params()
        return params


    def make_model(self) -> Experiment:
        self.logger.info("Instantiating model for experiment.")
        # pylint: disable=attribute-defined-outside-init
        self.model = SGDSIPDeconvolution1D(
            start = -10,
            end = 10,
            step = 1E-1,
            kernel = lambda x: np.float64(x >= 0),
            initial_guess = lambda x: np.zeros_like(x),
            learning_rate = lambda i: np.float64(1/np.sqrt(i)),
            loss = losses.SquaredLoss(),
            fixed_learning_rate = True,
        )
        self.has_model = True
        self.params = self.get_experiment_params_dict()


    def fit_model(
        self,
        features: np.ndarray,
        target: np.ndarray = None,
        force_refit: bool = False
    ) -> None:
        """Fit model."""
        assert self.has_model
        model_path = self.find_fitted_model()
        if model_path is None or force_refit:
            self.logger.info("Fitting model.")
            self.model.fit(features, target)
            self.logger.info("Model fitted.")
        else:
            self.logger.info("Loading pre-trained model.")
            self.model = pickle.load(model_path)
        self.is_fitted = True


    def make_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create data for experiment."""
        data = synthetic_data.make_noised_convolution(
            start = -10,
            end = 10,
            func = lambda w: np.exp(-w**2),
            kernel = lambda x: np.float64(x >= 0),
            step_gen = 1E-2,
            # step_obs = 1E-1,
            noise_var = 0.01,
            random_observations = True,
            n_samples = 1000
        )
        return data


    def make_target_plot(self, plots_path: Path):
        """Creates plot like the article"s."""
        fig, ax = plt.subplots()
        plot_x = self.model.discretized_domain
        ax.plot(plot_x, self.model.estimate_, "b-", label="Estimate")
        ax.plot(plot_x, np.exp(-plot_x**2), "r--", label="Truth")
        ax.set_xlabel("w")
        ax.legend()
        fig.savefig(plots_path / "truth_and_estimate.pdf")


    def make_observation_plot(
        self,
        x: np.ndarray,
        obs_y: np.ndarray,
        true_y: np.ndarray,
        plots_path: Path,
    ) -> None:
        """Make plot of the observed data"""
        fig, ax = plt.subplots()
        idx_sorted = np.argsort(x)
        x = x[idx_sorted]
        obs_y = obs_y[idx_sorted]
        true_y = true_y[idx_sorted]
        ax.plot(x, obs_y, "r.", label="With noise (data points)")
        ax.plot(x, true_y, "b-", label="Without noise")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        fig.savefig(plots_path / "data_points.pdf")


    def make_plots(
        self,
        x: np.ndarray,
        obs_y: np.ndarray,
        true_y: np.ndarray,
    ) -> None:
        """Make all plots"""
        plots_path = Path("./fig")
        self.logger.debug("Creating folder for plots.")
        plots_path.mkdir(exist_ok=True)

        self.make_target_plot(plots_path)
        self.make_observation_plot(x, obs_y, true_y, plots_path)


    def run(self) -> None:
        """Run the experiment"""
        run_path = self.setup_run()
        if not self.has_model:
            self.make_model()
        with InPath(run_path):
            self.save_params_dict()
            X, y_noised, y_true = self.make_data()
            self.fit_model(X, y_noised)
            self.save_model()
            self.make_plots(X, y_noised, y_true)
