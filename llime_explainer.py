import utils
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Any
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist


def distance(reference_vec: np.ndarray, neighbor_matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine distances between a reference vector and a matrix of vectors.

    This function is used to measure similarity between the original sentence
    and its generated neighbors in the interpretable feature space.

    Args:
        reference_vec (np.ndarray):
            Feature vector representing the reference sentence.
            Shape: (num_features,).
        neighbor_matrix (np.ndarray):
            Feature matrix representing neighboring sentences.
            Shape: (num_neighbors, num_features).

    Returns:
        np.ndarray:
            A 1D array of cosine distances where:
            - 0 indicates identical vectors
            - 1 indicates orthogonality
            Shape: (num_neighbors,).
    """
    # Ensure reference_vec is 2D for compatibility with scipy.cdist
    ref = reference_vec.reshape(1, -1)

    # Cosine distance = 1 - cosine similarity
    return cdist(ref, neighbor_matrix, metric='cosine').flatten()


def kernel_exp(d: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    Applies an exponential kernel to distance values.

    This kernel assigns higher weights to points closer to the reference
    sentence and lower weights to distant neighbors.

    Args:
        d (np.ndarray):
            Array of distance values.
        sigma (float, optional):
            Kernel width parameter controlling locality.
            Smaller values emphasize closer neighbors.
            Defaults to 0.5.

    Returns:
        np.ndarray:
            Kernel weights corresponding to each distance.
            Shape matches the input distance array.
    """
    # Add epsilon to prevent division-by-zero or numerical instability
    return np.sqrt(np.exp(-(d ** 2) / (sigma ** 2 + 1e-9)))


class LLiMeExplainer:
    """
    LLiMe.

    This class provides local explanations for black-box classifiers
    operating on text by:
    - Generating semantically meaningful neighbors using LLMs
    - Training a locally weighted linear surrogate
    - Extracting influential tokens for interpretability
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        one_word_neighs_prompt_path: str,
        neighs_prompt_path: str,
        counterfactual_system_prompt_path: str,
        local_surrogate = LogisticRegression()
    ):
        """
        Initializes the LLiMe explainer and its LLM-based generator.

        Args:
            model_name (str):
                Name of the LLM used for neighbor and counterfactual generation.
            api_key (str):
                API key for accessing the OpenRouter service.
            one_word_neighs_prompt_path (str):
                Path to the system prompt for one-word neighbor generation.
            neighs_prompt_path (str):
                Path to the system prompt for iterative neighbor generation.
            counterfactual_system_prompt_path (str):
                Path to the system prompt for counterfactual generation.

        Returns:
            None
        """
        # Deferred import avoids circular dependencies
        from generator import Generator

        self.generator = Generator(
            model_name,
            api_key,
            one_word_neighs_prompt_path,
            neighs_prompt_path,
            counterfactual_system_prompt_path
        )

        self.local_surrogate = local_surrogate

    def get_local_surrogate(
        self,
        sentence: str,
        classes: List[str],
        classifier_fn: Callable,
        n: int,
        max_runs: int
    ) -> Tuple[
        LogisticRegression,
        Dict[int, str],
        pd.DataFrame,
        np.ndarray,
        List[str]
    ]:
        """
        Builds a locally faithful surrogate model around a target sentence.

        Args:
            sentence (str):
                Input sentence to be explained.
            classes (List[str]):
                List of class names (binary classification assumed).
            classifier_fn (Callable):
                Black-box classifier function that maps sentences
                to class probability vectors.
            n (int):
                Desired number of neighbors per class.

        Returns:
            Tuple containing:
                - LogisticRegression:
                    Trained locally weighted surrogate model.
                - Dict[int, str]:
                    Mapping from feature index to token (vocabulary).
                - pd.DataFrame:
                    Binary feature matrix for original and neighboring sentences.
                - np.ndarray:
                    Predicted labels for each sentence (argmax of classifier output).
                - List[str]:
                    All sentences used (original + neighbors).
        """
        # 1. Generate LLM-based neighbors near the decision boundary
        neighbors = self.generator.neighbors_generation(
            sentence,
            n,
            classes,
            classifier_fn,
            max_runs
        )

        all_sentences = [sentence] + neighbors

        # 2. Build vocabulary from all tokens appearing in the local neighborhood
        all_tokens = []
        for s in all_sentences:
            all_tokens.extend(utils.split_sentence(s))

        features = list(set(all_tokens))

        # 3. Construct binary feature matrix (bag-of-words)
        # Each row corresponds to a sentence
        data = [
            [1 if token in s else 0 for token in features]
            for s in all_sentences
        ]

        x = pd.DataFrame(data, columns=features)

        # 4. Compute distances, weights, and labels
        dists = distance(x.iloc[0].values, x.values)
        weights = kernel_exp(dists)

        # Black-box classifier predictions
        y = classifier_fn(all_sentences).argmax(axis=1)

        # 5. Train locally weighted logistic regression surrogate
        local_surrogate = self.local_surrogate

        local_surrogate.fit(x, y, sample_weight=weights)

        vocabulary = {i: token for i, token in enumerate(features)}

        return local_surrogate, vocabulary, x, y, all_sentences

    def get_explaination(
        self,
        logreg: LogisticRegression,
        vocabulary: Dict[int, str],
        x: pd.DataFrame,
        label: int,
        p: float
    ) -> Dict[str, Any]:
        """
        Extracts influential tokens from the trained surrogate model.

        Tokens are selected based on coefficient magnitude and logical
        consistency with the predicted class.

        Args:
            logreg (LogisticRegression):
                Trained local surrogate model.
            vocabulary (Dict[int, str]):
                Mapping from feature index to token.
            x (pd.DataFrame):
                Feature matrix used to train the surrogate.
            label (int):
                Predicted label of the original sentence.
            p (float):
                Percentile threshold (0 < p < 1) controlling sparsity
                of the explanation.

        Returns:
            Dict[str, Any]:
                Dictionary with:
                - "tokens": List of influential tokens
                - "coeff": Corresponding surrogate coefficients
        """
        coeffs = logreg.coef_[0]

        # Separate positive and negative coefficients
        pos_coeffs = coeffs[coeffs > 0]
        neg_coeffs = coeffs[coeffs < 0]

        # Compute percentile-based thresholds
        c_neg_thresh = (
            np.percentile(neg_coeffs, p * 100)
            if neg_coeffs.size > 0 else -float('inf')
        )

        c_pos_thresh = (
            np.percentile(pos_coeffs, (1 - p) * 100)
            if pos_coeffs.size > 0 else float('inf')
        )

        tokens_to_save = []
        coeff_to_save = []

        # Iterate over all features
        for i, c in enumerate(coeffs):
            is_present = x.iloc[0, i] == 1

            # Skip weak (non-informative) coefficients
            if c_neg_thresh < c < c_pos_thresh:
                continue

            # Logical pruning:
            # Keep only features that meaningfully explain the prediction
            if c < 0:
                if (label == 0 and not is_present) or (label == 1 and is_present):
                    continue
            elif c > 0:
                if (label == 1 and not is_present) or (label == 0 and is_present):
                    continue

            tokens_to_save.append(vocabulary[i])
            coeff_to_save.append(float(c))

        return {
            "tokens": tokens_to_save,
            "coeff": coeff_to_save
        }

    def get_counterfactual(
        self,
        sentence: str,
        explanation: Dict[str, Any],
        label: int,
        classes: List[str],
        classifier_fn: Callable,
        max_runs: int
    ):
        """
        Generates a counterfactual explanation for the input sentence.

        Args:
            sentence (str):
                Original sentence to be perturbed.
            explanation (Dict[str, Any]):
                Explanation dictionary produced by `get_explaination`.
            label (int):
                Original predicted class label.
            classes (List[str]):
                List of class names.
            classifier_fn (Callable):
                Black-box classifier function.
            max_runs (int):
                Maximum number of LLM attempts.

        Returns:
            Optional[Tuple[str, Any]]:
                Counterfactual sentence and its explanation,
                or None if generation fails.
        """
        return self.generator.counterfactual_generation(
            sentence,
            explanation,
            label,
            classes,
            classifier_fn,
            max_runs
        )
