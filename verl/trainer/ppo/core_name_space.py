from enum import Enum

__all__ = ["AdvantageEstimator", "LossCalculator"]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    DCPO = "dcpo"
    DAPO = "dapo"


class LossCalculator(str, Enum):
    """
    support loss calculator types
    """

    TM = "token-mean"
    OTM = "only-token-mean"
    SMTS = "seq-mean-token-sum"
    SMTM = "seq-mean-token-mean"
    SMTSN = "seq-mean-token-sum-norm"
