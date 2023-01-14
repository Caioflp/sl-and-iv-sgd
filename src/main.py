"""Experiments are run from here"""

from src.experiments import ArticleDeconvolutionExample
from src.utils.logs import setup_main_logger

logger = setup_main_logger()


if __name__ == "__main__":
    exp = ArticleDeconvolutionExample()
    exp.run()
