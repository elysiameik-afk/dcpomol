# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward function for molecule generation task.
Evaluates generated molecules based on multiple drug-like properties.
"""

import re
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    from rdkit.Chem.Crippen import MolLogP
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Molecule generation reward will not work.")

# Try to import SA_Score
SA_SCORE_AVAILABLE = False
if RDKIT_AVAILABLE:
    try:
        from rdkit.Chem import RDConfig
        import sys
        sys.path.append(f'{RDConfig.RDContribDir}/SA_Score')
        import sascorer
        SA_SCORE_AVAILABLE = True
    except Exception as e:
        logger.warning(f"SA_Score not available: {e}")


def check_format(response_str: str) -> Tuple[float, Optional[str]]:
    """
    Check if the response has correct <SMILES>...</SMILES> format.
    
    Args:
        response_str: Model's response string
        
    Returns:
        (format_reward, smiles_string or None)
        format_reward: +1.0 if correct, -5.0 if incorrect
    """
    # Find all <SMILES> and </SMILES> tags
    open_tags = re.findall(r'<SMILES>', response_str, re.IGNORECASE)
    close_tags = re.findall(r'</SMILES>', response_str, re.IGNORECASE)
    
    # Must have exactly one pair in correct order
    if len(open_tags) == 1 and len(close_tags) == 1:
        match = re.search(r'<SMILES>\s*(.+?)\s*</SMILES>', response_str, re.IGNORECASE)
        if match:
            smiles = match.group(1).strip()
            if smiles:  # Non-empty SMILES
                return 1.0, smiles
    
    return -5.0, None


def check_validity(smiles: str) -> Tuple[float, Optional[object]]:
    """
    Check if SMILES string is a valid molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        (validity_reward, mol_object or None)
        validity_reward: +0.5 if valid, -3.0 if invalid
    """
    if not RDKIT_AVAILABLE:
        return -3.0, None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -3.0, None
        return 0.5, mol
    except Exception:
        return -3.0, None


def qed_reward(mol) -> float:
    """
    QED (Quantitative Estimate of Drug-likeness) reward.
    Range: 0-2.0
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        QED reward (0-2.0)
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    try:
        qed_score = QED.qed(mol)
        # Map 0.5-1.0 to 0-2.0
        if qed_score < 0.5:
            return 0.0
        return (qed_score - 0.5) * 4.0
    except Exception:
        return 0.0


def sa_reward(mol) -> float:
    """
    SA Score (Synthetic Accessibility) reward.
    Range: 0-2.0 (lower SA is better)
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        SA reward (0-2.0)
    """
    if not SA_SCORE_AVAILABLE:
        # If SA_Score not available, give neutral score
        return 1.0
    
    try:
        sa_score = sascorer.calculateScore(mol)
        # SA range: 1-10, lower is better
        # Map: 1-4 → 2.0-0.5, 4-7 → 0.5-0, >7 → 0
        if sa_score <= 4.0:
            return 2.0 - (sa_score - 1.0) * 0.5
        elif sa_score <= 7.0:
            return max(0.0, 0.5 - (sa_score - 4.0) * 0.15)
        else:
            return 0.0
    except Exception:
        return 0.0


def logp_reward(mol) -> float:
    """
    LogP (lipophilicity) reward.
    Range: 0-1.5
    Ideal range: 1-4, peak at 2.5
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        LogP reward (0-1.5)
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    try:
        logp = MolLogP(mol)
        
        if 1.0 <= logp <= 4.0:
            # Peak at 2.5
            return 1.5 - abs(logp - 2.5) * 0.3
        elif 0 <= logp < 1.0:
            return logp * 0.8
        elif 4.0 < logp <= 5.0:
            return max(0.0, 0.5 - (logp - 4.0) * 0.3)
        else:
            return 0.0
    except Exception:
        return 0.0


def mw_reward(mol) -> float:
    """
    Molecular Weight reward.
    Range: 0-1.0
    Ideal range: 200-500
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        MW reward (0-1.0)
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    try:
        mw = Descriptors.MolWt(mol)
        
        if 200 <= mw <= 500:
            return 1.0
        elif 150 <= mw < 200:
            return (mw - 150) / 50 * 0.5
        elif 500 < mw <= 600:
            return max(0.0, 1.0 - (mw - 500) / 100)
        else:
            return 0.0
    except Exception:
        return 0.0


def tpsa_reward(mol) -> float:
    """
    TPSA (Topological Polar Surface Area) reward.
    Range: 0-1.0
    Ideal range: 40-120
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        TPSA reward (0-1.0)
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    try:
        tpsa = Descriptors.TPSA(mol)
        
        if 40 <= tpsa <= 120:
            return 1.0
        elif 20 <= tpsa < 40:
            return (tpsa - 20) / 20 * 0.5
        elif 120 < tpsa <= 140:
            return max(0.0, 1.0 - (tpsa - 120) / 20)
        else:
            return 0.0
    except Exception:
        return 0.0


def lipinski_reward(mol) -> float:
    """
    Lipinski's Rule of Five reward.
    Range: 0-1.5
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Lipinski reward (0-1.5)
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        
        # 0 violations → 1.5, 1 → 1.0, 2 → 0.5, ≥3 → 0
        if violations == 0:
            return 1.5
        elif violations == 1:
            return 1.0
        elif violations == 2:
            return 0.5
        else:
            return 0.0
    except Exception:
        return 0.0


def egfr_specific_reward(mol) -> float:
    """
    EGFR inhibitor-specific structural features reward.
    Range: 0-1.0
    
    Features:
    - Aromatic rings: 2-4 ideal
    - Rotatable bonds: 5-10 ideal
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        EGFR-specific reward (0-1.0)
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    try:
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        score = 0.0
        
        # Aromatic rings reward (0-0.6)
        if 2 <= aromatic_rings <= 4:
            score += 0.6
        elif aromatic_rings == 1 or aromatic_rings == 5:
            score += 0.3
        
        # Rotatable bonds reward (0-0.4)
        if 5 <= rotatable_bonds <= 10:
            score += 0.4
        elif 3 <= rotatable_bonds < 5 or 10 < rotatable_bonds <= 12:
            score += 0.2
        
        return score
    except Exception:
        return 0.0


def compute_score(solution_str: str, ground_truth: str = "") -> Dict:
    """
    Compute comprehensive reward for generated molecule.
    
    Score components:
    - Format: +1.0 (correct) / -5.0 (incorrect)
    - Validity: +0.5 (valid) / -3.0 (invalid)
    - QED: 0-2.0
    - SA: 0-2.0
    - LogP: 0-1.5
    - MW: 0-1.0
    - TPSA: 0-1.0
    - Lipinski: 0-1.5
    - EGFR: 0-1.0
    
    Total range: -5.0 (worst) to ~10.0 (best)
    
    Args:
        solution_str: Model's response string
        ground_truth: Not used for molecule generation (kept for API compatibility)
        
    Returns:
        Dictionary with:
        - score: total reward
        - acc: whether molecule is valid and drug-like (score > 3.0)
        - pred: extracted SMILES or "[INVALID]"
        - Individual component scores for analysis
    """
    if not RDKIT_AVAILABLE:
        return {
            "score": -5.0,
            "acc": False,
            "pred": "[RDKIT_NOT_AVAILABLE]",
            "format": -5.0
        }
    
    reward_dict = {}
    
    # 1. Format check
    format_reward, smiles = check_format(solution_str)
    reward_dict["format"] = format_reward
    
    if smiles is None:
        return {
            "score": format_reward,
            "acc": False,
            "pred": "[INVALID_FORMAT]",
            **reward_dict
        }
    
    # 2. Validity check
    validity_reward, mol = check_validity(smiles)
    reward_dict["validity"] = validity_reward
    
    if mol is None:
        total_score = format_reward + validity_reward
        return {
            "score": total_score,
            "acc": False,
            "pred": smiles,  # Return the invalid SMILES for debugging
            **reward_dict
        }
    
    # 3. Property rewards
    reward_dict["qed"] = qed_reward(mol)
    reward_dict["sa"] = sa_reward(mol)
    reward_dict["logp"] = logp_reward(mol)
    reward_dict["mw"] = mw_reward(mol)
    reward_dict["tpsa"] = tpsa_reward(mol)
    reward_dict["lipinski"] = lipinski_reward(mol)
    reward_dict["egfr"] = egfr_specific_reward(mol)
    
    # 4. Total score
    total_score = (
        format_reward +
        validity_reward +
        reward_dict["qed"] +
        reward_dict["sa"] +
        reward_dict["logp"] +
        reward_dict["mw"] +
        reward_dict["tpsa"] +
        reward_dict["lipinski"] +
        reward_dict["egfr"]
    )
    
    # 5. Accuracy: consider molecule "good" if score > 3.0
    # (meaning it's valid and has decent drug-like properties)
    acc = total_score > 3.0
    
    return {
        "score": total_score,
        "acc": acc,
        "pred": smiles,
        **reward_dict
    }

